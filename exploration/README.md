# Model structure exploration

Inspect the pinned `guidelabs/steerling-8b-instruct` architecture without
generating text or modifying the model.

Install the optional dependencies:

```bash
python -m pip install -e '.[model-inspection]'
python -m pip install 'steerling @ git+https://github.com/guidelabs/steerling.git@f34ffa89e46969445f3cf6e7c885e9623a2047c1'
```

The official Steerling package currently requires Python 3.13.

The default mode downloads only configuration and remote model source files:

```bash
python exploration/print_model_structure.py
```

Build the complete module hierarchy on PyTorch's meta device (no weight or VRAM
allocation) and print layers, module types, parameter shapes, and estimated
BF16/FP32 sizes:

```bash
python exploration/print_model_structure.py --mode meta --max-depth 5
```

To inspect actual loaded dtypes and devices, use a machine with enough memory:

```bash
python exploration/print_model_structure.py --mode loaded --device-map auto
```

The model uses trusted remote code pinned to the immutable revision recorded in
the script. Review that revision before changing it.
