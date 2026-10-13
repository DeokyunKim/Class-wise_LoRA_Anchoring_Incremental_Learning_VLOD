import torch

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class TaskMemory:
    def __init__(self):
        self._current_classes = []
        self._modules = {}
        self._memory = {}
        """
        self._modules = {
            layer_id: {
                class_name: [tensor_A, tensor_B]
            }
        }
        """
        self._registered_instances = []
        self._class_counter = {}
        self.current_task = None
        self.is_warm_up = True
        self.task_index = 0
        self.task_mapping = {}   # dataset -> {class_idx: class_name}; set in main() before use
        self.lambda_b = 0.7      # DitHub cross-task shared-B merge coefficient (set from --lambda-b)
        # ── Class-wise LoRA Anchoring (per-class L2 anchor) ──────────────────
        # At each task start (task_index>0) snapshot the selected LoRA params and,
        # during training, penalize ||θ_LoRA - θ_snapshot||² so the per-class
        # inventory does not drift away from prior-task knowledge.
        self.l2_anchor_weight = 0.0        # γ_L2 (0 = disabled = vanilla DitHub)
        self.l2_anchor_target = "a_and_b"  # 'a_only' | 'a_and_b'
        self._l2_anchor_dict = {}          # param_name -> snapshot tensor (refreshed per task)
        # ── Optional inter-class repulsion on per-class A (helps the class-overlap regime) ──
        self.repulsion_weight = 0.0          # γ_rep (0 = disabled)
        self.repulsion_margin = 1.0
        self.repulsion_pos_weight = 1.0      # α on positive term; 0 = repulsion-only (pos ≡ L2 anchor, removable)
        self._repulsion_anchor = {}          # id(module) -> {class_key: A_c snapshot}

    @torch.no_grad()
    def l2_anchor_snapshot(self, model, target="a_and_b",
                           skip_warmup=False, skip_new_per_class=False):
        """Snapshot selected LoRA params at task start (refresh, not accumulate).
        target ∈ {'a_only','a_and_b'}:
          'a_only'  = anchor only A-type params (per_class_lora_A + warmup_lora_a);
                      shared_lora_b left free (stock DitHub cross-task merge).
          'a_and_b' = anchor per-class A (+ warmup A) AND shared_lora_b.
        skip_warmup: exclude warmup_lora_a from the anchor (warmup module free to adapt — new-class
                     warm-up trains DitHub-style without anchor drag).
        skip_new_per_class: exclude per-class A of classes never trained yet (class_counter==0) so
                     genuinely NEW classes adapt freely (DitHub-style); re-appearing classes
                     (counter>0) stay anchored = preserved. (CLA-selective.)
        Returns the number of params snapshotted.
        """
        self._l2_anchor_dict = {}
        skip_class_keys = set()
        if skip_new_per_class:
            for c, cnt in self._class_counter.items():
                if cnt == 0:
                    skip_class_keys.add(f"class_{c}")
        n = 0
        for name, p in model.named_parameters():
            if not (p.requires_grad and "lora_" in name):
                continue
            if target == "a_only" and "shared_lora_b" in name:
                continue
            if skip_warmup and "warmup_lora" in name:
                continue
            if skip_class_keys and any(k in name for k in skip_class_keys):
                continue
            self._l2_anchor_dict[name] = p.detach().clone()
            n += 1
        return n

    def l2_anchor_loss(self, model):
        """Σ_p ||p - anchor_p||² / n  over snapshotted params (per-param normalized)."""
        if not self._l2_anchor_dict:
            return torch.zeros((), device=next(model.parameters()).device)
        total = 0.0
        n = 0
        for name, p in model.named_parameters():
            if name in self._l2_anchor_dict and p.requires_grad:
                anchor = self._l2_anchor_dict[name]
                if anchor.shape == p.shape:
                    total = total + (p - anchor.to(p.device)).pow(2).sum()
                    n += 1
        if n == 0:
            return torch.zeros((), device=next(model.parameters()).device)
        return total / n

    @torch.no_grad()
    def repulsion_anchor_snapshot(self, model):
        """Snapshot per_class_lora_A per module at task start (positive anchor)."""
        self._repulsion_anchor = {}
        n = 0
        for module in model.modules():
            if not hasattr(module, "per_class_lora_A"):
                continue
            self._repulsion_anchor[id(module)] = {
                c: module.per_class_lora_A[c].detach().clone()
                for c in module.per_class_lora_A
            }
            n += 1
        return n

    def repulsion_loss_per_class_A(self, model, margin=1.0, pos_weight=1.0):
        """Margin-based inter-class repulsion on per-class A: per module, averaged over layers.
        Pushes distinct per-class A apart while (optionally) holding each near its prior anchor.
          pos_d = ||A_c - A_c_anchor||²_mean   (anchor detached)
          neg_d = mean_{c'≠c} ||A_c - A_{c'}.detach()||²_mean   (sibling detached)
          L_c   = mean_c max(0, pos_weight·pos_d - neg_d + margin)
        Gram identity avoids the (C,C,r,d) intermediate."""
        if not self._repulsion_anchor:
            return None
        total = None
        n_layers = 0
        for module in model.modules():
            if not hasattr(module, "per_class_lora_A"):
                continue
            anchor = self._repulsion_anchor.get(id(module))
            if not anchor:
                continue
            classes = [c for c in anchor if c in module.per_class_lora_A]
            if len(classes) < 2:
                continue
            try:
                A_stack = torch.stack([module.per_class_lora_A[c] for c in classes])
            except RuntimeError:
                continue
            anchor_stack = torch.stack([anchor[c] for c in classes])
            pos = ((A_stack - anchor_stack) ** 2).mean(dim=(1, 2))      # (C,)
            A_flat = A_stack.flatten(1)
            A_flat_det = A_flat.detach()
            denom = float(A_flat.shape[1])
            sq_grad = (A_flat ** 2).sum(dim=1)
            sq_det = sq_grad.detach()
            gram_pn = A_flat @ A_flat_det.t()
            pair_d = (sq_grad.unsqueeze(1) + sq_det.unsqueeze(0) - 2.0 * gram_pn) / denom
            C = pair_d.shape[0]
            off = ~torch.eye(C, dtype=torch.bool, device=pair_d.device)
            neg = pair_d.masked_select(off).view(C, C - 1).mean(dim=1)  # (C,)
            repulsion = torch.clamp(pos_weight * pos - neg + margin, min=0).mean()
            total = repulsion if total is None else total + repulsion
            n_layers += 1
        if total is None or n_layers == 0:
            return None
        return total / n_layers

    def get_classes(self):
        assert self._current_classes, 'Please set self._current_classes before getting them'
        return self._current_classes

    def set_classes(self, classes):
        self._current_classes = classes

    def get_modules(self, layer_id):
        if layer_id not in self._modules:
            return {}
        return self._modules[layer_id]

    def set_counter(self, classes):
        if not self._class_counter:
            for c in classes:
                self._class_counter[c] = 0

    def add_counter(self, classes):
        if self.is_warm_up:
            return
        for c in classes:
            self._class_counter[c[6:]] += 1

    def print_counter(self):
        for key in sorted(self._class_counter.keys()):
            print(f"{key}: {self._class_counter[key]}")

    @torch.no_grad()
    def save_modules(self):
        for m in self._registered_instances:
            if m.layer_name not in self._modules:
                self._modules[m.layer_name] = {}
            curr_classes = m.per_class_lora_A.keys()
            for curr_class in curr_classes:
                a_b = [
                    m.per_class_lora_A[curr_class]
                ]
                self._modules[m.layer_name][curr_class] = a_b
                self.set_memory("B_shared", m.layer_name, m.shared_lora_b)
        self._registered_instances = []

    @torch.no_grad()
    def merge_B(self):
        if any(k.startswith("B_prev_task") for k in self._memory):
            lambda_b_marouf = self.lambda_b
            for key in self._memory.keys():
                if key.startswith("B_shared"):
                    suffix = key[len("B_shared"):]
                    prev_task_key = "B_prev_task" + suffix
                    assert prev_task_key in self._memory, f"{prev_task_key} non trovato in self._memory"
                    self._memory[key] = (1 - lambda_b_marouf) * self._memory[prev_task_key] + lambda_b_marouf * self._memory[key]

    def set_memory(self, key, layer_name, value):
        assert isinstance(value, torch.Tensor), f"Value must be a tensor, but got {type(value)}"
        self._memory[key+layer_name] = value

    def get_memory(self, key, layer_name):
        return self._memory.get(key+layer_name, None)

    def end_task(self):
        self.save_modules()
        self.print_counter()
        self.merge_B()
        self.is_warm_up = True
        self.task_index += 1

    def enable_per_class(self):
        self.is_warm_up = False
        for m in self._registered_instances:
            m.enable_per_class(self.current_task)

    def register_module(self, instance):
        self._registered_instances.append(instance)
