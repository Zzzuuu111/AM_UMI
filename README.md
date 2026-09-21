# lerobot_alohamini

Shared software layer for the AlohaMini product line, built on HuggingFace LeRobot. Supports both the full AlohaMini robot (dual-arm + mobile base + lift) and the AM-ARM200 arm.

> Haven't assembled your hardware yet? Start here: [AlohaMini](https://github.com/liyiteng/alohamini) · [AM-ARM200](https://github.com/liyiteng/AM-ARM)

## Updates
- **[2025-07-11]** merge upstream LeRobot v0.6

## Documentation

Start with setup, then follow the workflow for your hardware. Use the reference pages when you need exact flags, commands, or low-level debug tools.

### Recommended Path

1. [Install](docs/alohamini/install.md) — prepare the environment, serial port permissions, and Hugging Face login.
2. Pick your robot workflow:
   - [AM-ARM200](docs/alohamini/am-arm200.md) — single-arm workflow on one PC: calibration, teleoperation, dataset recording, training, and evaluation.
   - [AlohaMini 1 / 2 / 2 Pro](docs/alohamini/alohamini.md) — dual-arm workflow with Pi + PC: calibration, teleoperation, dataset recording, training, and evaluation.

### References

| Reference | Use it for |
|-----------|------------|
| [Hardware Profiles](docs/alohamini/profiles.md) | `--arm_profile` and `--robot_model` flag meanings |
| [Command Cheat Sheet](docs/alohamini/commands.md) | Copy-paste commands for setup, host, teleoperation, recording, training, evaluation, and common checks |
| [Debug Tools](examples/debug/README.md) | Low-level motor, wheel, lift axis, servo ID, phase, midpoint, torque, and scripted-action debug functions |

---

## Team & Contact

AlohaMini is created by **Li Yiteng** and **Wu Zhiyong**.

- Email: liyiteng+github@gmail.com
- WeChat: liyiteng

## Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) — the software stack this repository targets
- [ALOHA](https://tonyzhaozh.github.io/aloha/) — the bimanual teleoperation paradigm
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) — pioneered the low-cost open arm design pattern
