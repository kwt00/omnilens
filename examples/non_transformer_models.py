"""omnilens works on ANY PyTorch model — not just transformer LLMs.

This example demonstrates using omnilens with:
1. Mamba (state space model)
2. RWKV (linear attention alternative)
3. Custom models (world models, vision models, etc.)
"""

import torch
from omnilens import TappedModel

# =============================================================================
# Example 1: Mamba — State Space Model
# =============================================================================

print("=" * 60)
print("Mamba (SSM)")
print("=" * 60)

model = TappedModel.from_pretrained("state-spaces/mamba-130m-hf", dtype=torch.float32)
print(model)

# Cache mixer components — Mamba has no attention, just mixer blocks
logits, cache = model.run_with_cache(
    text="The quick brown fox",
    names=[
        "layers.0.mixer.layer_norm",
        "layers.0.mixer.in_proj",
        "layers.0.mixer.conv1d",
        "layers.0.mixer.out_proj",
        "layers.12.mixer.in_proj",
        "layers.12.mixer.out_proj",
    ],
)

print("\nCached activations:")
for name, tensor in cache.items():
    print(f"  {name}: {tensor.shape}")

# Intervene on Mamba's mixer
baseline_logits, _ = model.run_with_cache(text="The quick brown fox")


def zero_mixer_output(activation, hook_name):
    return torch.zeros_like(activation)


modified_logits = model.run_with_hooks(
    text="The quick brown fox",
    hooks={"layers.6.mixer.out_proj": zero_mixer_output},
)

print(f"\nIntervention changed output: {not torch.allclose(baseline_logits, modified_logits)}")
del model


# =============================================================================
# Example 2: RWKV — Linear Attention Alternative
# =============================================================================

print("\n" + "=" * 60)
print("RWKV (Linear Attention)")
print("=" * 60)

model = TappedModel.from_pretrained("RWKV/rwkv-4-169m-pile", dtype=torch.float32)
print(model)

# RWKV uses time_mix (attention-like) and channel_mix (feed-forward-like)
logits, cache = model.run_with_cache(
    text="Hello world",
    names=[
        "layers.0.time_mix.key",
        "layers.0.time_mix.value",
        "layers.0.time_mix.receptance",
        "layers.0.time_mix.output",
        "layers.0.channel_mix.key",
        "layers.0.channel_mix.value",
    ],
)

print("\nCached activations:")
for name, tensor in cache.items():
    print(f"  {name}: {tensor.shape}")

del model


# =============================================================================
# Example 3: Any Custom PyTorch Model
# =============================================================================

print("\n" + "=" * 60)
print("Custom PyTorch Model (e.g. world model, vision model)")
print("=" * 60)


# Define a toy model — could be LeWorldModel, a ViT, a diffusion model, etc.
class ToyWorldModel(torch.nn.Module):
    def __init__(self, d_state=64, d_action=4, n_layers=3):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(d_state, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )
        self.predictor = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(64 + d_action, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 64),
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder = torch.nn.Linear(64, d_state)

    def forward(self, state, action):
        latent = self.encoder(state)
        combined = torch.cat([latent, action], dim=-1)
        for layer in self.predictor:
            combined = layer(combined)
            combined = torch.cat([combined, action], dim=-1)
        return self.decoder(combined[:, :64])


# Wrap with TappedModel — no registry needed, use raw module names
raw_model = ToyWorldModel()
model = TappedModel(model=raw_model)

print("Module tree:")
model.print_module_tree()

# Cache activations using raw module names
state = torch.randn(1, 64)
action = torch.randn(1, 4)

# Hook and cache specific components
hook_names = [
    "encoder.0",  # first linear layer of encoder
    "encoder.2",  # second linear layer of encoder
    "predictor.0.0",  # first linear of first predictor layer
    "predictor.2.2",  # last linear of last predictor layer
    "decoder",  # final decoder
]

# For custom models, use run_with_hooks directly since run_with_cache
# expects standard model signatures. The hook primitives work on anything:
cached = {}
handles = []
for name in hook_names:
    parts = name.split(".")
    module = raw_model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    def make_hook(hook_name):
        def hook_fn(mod, inp, out):
            cached[hook_name] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
        return hook_fn

    handles.append(module.register_forward_hook(make_hook(name)))

# Run forward pass
output = raw_model(state, action)

# Clean up hooks
for h in handles:
    h.remove()

print("\nCached activations from custom world model:")
for name, tensor in cached.items():
    print(f"  {name}: {tensor.shape}")

print("\n" + "=" * 60)
print("omnilens: any model, any architecture, any modality")
print("=" * 60)
