# Model and research attribution

Original HOI4-AI code is Copyright (c) 2026 HOI4-AI contributors and licensed under MIT OR Apache-2.0. This grant does not change third-party terms below.

The visual encoder uses [LeVJEPA-VideoMix-Large](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large), revision `e831a0347737fcaa660b39c57d41c109de399845`, published by galilai-group. Its model card specifies **CC BY-NC 4.0**. Preserve upstream attribution and terms when using or sharing weights and derived models. Downloaded weights and custom model code are not committed here.

Sparse predictive learning is adapted from [LpWM's official implementation](https://github.com/YilunKuang/lpworldmodel), by Yilun Kuang and collaborators, published under the MIT license. This prototype implements RepReLU and rectified-Laplace random-projection regularization in its own policy training code. It uses fewer projections and a HOI4-specific recurrent action interface; it is not an official implementation or a reproduction claim.

LpWM's upstream license notice is Copyright (c) 2025 gaoyuezhou. The complete upstream notice and MIT terms are preserved in [third_party/licenses/lpworldmodel-MIT.txt](third_party/licenses/lpworldmodel-MIT.txt).

HOI4 is a Paradox game. The map generator reads palettes from the user's local installation and creates local training artifacts. Game assets and game binaries are not included in the source repository or the portable worker bundle.
