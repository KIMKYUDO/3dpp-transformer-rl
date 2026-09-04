# Transformer-Based Reinforcement Learning for 3D Packing

A Kyungpook National University team project on **3D container packing for reverse logistics**, exploring transformer-based policies and Proximal Policy Optimization (PPO).

## Research focus

How can a learned policy select, orient, and place items while improving container space utilization?

The project combines a packing simulator, transformer-based policy/value components, and reinforcement-learning training. Space utilization is a project objective, not a claim of independently verified improvement over every baseline.

## What is in this repository?

| Component | Location |
|---|---|
| Container simulation | [envs/container_sim.py](envs/container_sim.py) |
| Transformer and policy/value components | [agents/](agents/) |
| PPO training | [train/train_ppo.py](train/train_ppo.py) |
| Environment, model, and training settings | [configs/](configs/) |
| Benchmark entry point | [benchmark.py](benchmark.py) |
| Component tests | [tests/](tests/) |
| Proposal and project records | [docs/](docs/) |

This is a shared project codebase. Repository ownership alone should not be interpreted as sole authorship of all components.

## Getting started

1. Create an isolated Python environment.
2. Install the declared dependencies:

```bash
pip install -r requirements-cpu.txt
```

3. Review the settings in `configs/` and the training entry point before launching an experiment.
4. Inspect the benchmark and tests for the experiment-specific inputs and expected behavior.

The repository contains research code and a proposal. End-to-end training and benchmark results were not re-run during the documentation cleanup; no new performance claim is made here.

## Project records

- [Proposal and supporting documents](docs/)
- [Additional project notes on Notion](https://www.notion.so/returnall/2025-2-26adad4051a680a7bedfd65b457deed5?source=copy_link) — access may require permission.
- [Kyudo Kim's research overview](https://github.com/KIMKYUDO)
