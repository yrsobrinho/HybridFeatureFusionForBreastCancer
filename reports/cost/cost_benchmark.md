# Computational cost

Batch 16, 28x28, 4 qubits, 1 layer(s), patch 2. Simulator: `lightning.qubit` (CPU).

| Model | Device | Params (trainable) | MACs/img | Circuits/img | Inference ms/img | Train ms/step | Slowdown |
|---|---|---|---|---|---|---|---|
| shallow | cpu | 93,378 (93,378) | 425,344 | 0 | 0.03 | 2.5 | 1.0× |
| resnet | cpu | 11,171,266 (11,171,266) | 33,006,848 | 0 | 0.55 | 261.4 | 103.3× |
| quantum | cpu | 1,570 (1,570) | 1,568 | 196 | 1297.72 | 14849.5 | 5868.1× |
| extended | cpu | 1,574 (1,574) | 1,568 | 196 | 438.26 | 16534.6 | 6534.0× |
| shf-shallow | cpu | 210,888 (210,888) | 542,336 | 196 | 1055.62 | 65412.2 | 25849.2× |
| shf-resnet | cpu | 11,337,928 (11,337,928) | 33,172,224 | 196 | 635.31 | 10853.6 | 4289.1× |
| tshf-shallow | cpu | 210,889 (210,889) | 542,336 | 196 | 511.26 | 9857.5 | 3895.4× |
| tshf-resnet | cpu | 11,337,929 (11,337,929) | 33,172,224 | 196 | 591.56 | 12010.1 | 4746.1× |
| shallow | cuda | 93,378 (93,378) | 425,344 | 0 | 0.01 | 1.3 | 1.0× |
| resnet | cuda | 11,171,266 (11,171,266) | 33,006,848 | 0 | 0.07 | 4.6 | 3.5× |
| quantum | cuda | 1,570 (1,570) | 1,568 | 196 | 1174.34 | 16594.5 | 12485.2× |
| extended | cuda | 1,574 (1,574) | 1,568 | 196 | 551.12 | 15300.8 | 11511.8× |
| shf-shallow | cuda | 210,888 (210,888) | 542,336 | 196 | 637.77 | 11055.4 | 8317.7× |
| shf-resnet | cuda | 11,337,928 (11,337,928) | 33,172,224 | 196 | 666.39 | 16395.1 | 12335.1× |
| tshf-shallow | cuda | 210,889 (210,889) | 542,336 | 196 | 661.42 | 24320.0 | 18297.6× |
| tshf-resnet | cuda | 11,337,929 (11,337,929) | 33,172,224 | 196 | 1301.97 | 11233.4 | 8451.7× |

Median over timed repeats; `lightning.qubit` runs on CPU, so the quantum branch does not benefit from the GPU.
