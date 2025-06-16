**Superconducting Josephson Junction Reservoir Computing for Advanced Temporal Information Processing**


## Overview

With the advent of Artificial Intelligence (AI), the need
for a huge amount of dynamic data processing in a significant
amount of short time has become a regular necessity. Using
superconducting RC, we believe that an unprecedented milestone
can be achieved in terms of cost and speed. In this study,
we have simulated a reservoir computing system using
a Josephson Junction (JJ) crossbar array. Our simulated model
achieved competitive accuracy using only 81 JJs and training
only the output layer, where an advanced NN based model needs
to train both hidden layer and model’s output layer. Our results demonstrate that even a small-scale system containing just 81 JJs can successfully perform
tasks like handwritten digit recognition and cat pole task. After 794 training
iterations, the reservoir computing system gives 100% accuracy
to recognize all inputs from the provided data set. Moreover, while introduced with noisy digits, the model performs with 85.71% accuracy. 

## Background

The reservoir consists of recurrent, dynamically connected artificial neurons. In such frameworks, input layers are mapped to a higher-dimensional space within the reservoir layer, facilitating efficient signal recognition.
RC offers a significant advantage over conventional Neu-
ral Networks (NNs) by reducing computational complexity.
Unlike traditional NNs, where all connections and weights
are carefully structured and trained through backpropagation-
a time-consuming and computationally intensive process,
RC requires training only in the output layer. The reser-
voir layer’s weights remain fixed, eliminating the need for
backpropagation and drastically simplifying computation.
This approach not only reduces the computational burden
but also makes RC a cost-effective alternative to traditional
NNs. Additionally, the same reservoir can be repurposed for
multiple tasks by applying different sets of output weights,
enhancing both flexibility and efficiency. These characteristics
make RC particularly well-suited for applications involving
time-series analysis, pattern recognition, and other data-driven
tasks.
Various types of physical reservoirs have been explored,
including photonic circuits, spin devices, memristors, FPGAs, and even biological neurons. Among these, superconductors are particularly attractive because of their high-speed and low-power operation, making them a
promising choice for physical RC. Most of these hardware
platforms rely on a limited number of physical nonlinear
elements, combined with temporal multiplexing and feedback,
to introduce dimensionality into the reservoir. By converting input information
into quantized magnetic flux, the JTL facilitates signal propagation through the circuit in the form of spiking signals. 
Study showcased the JTL’s high-speed operation, achieving
100 Gb/s in channel equalization tasks, while also evaluating
its performance on fundamental tasks such as the parity task.
Superconducting reservoirs can be designed in various ways.
In this work, we focus on a crossbar JTL structure, as it allows
for a greater number of Josephson Junctions (JJs) to be integrated onto a single substrate. Unlike a linear structure, which
requires additional physical space and wiring for each connection, the crossbar configuration provides a more efficient
way to interconnect elements, facilitating better scalability and
increasing the potential for parallel processing. This crossbar
configuration enhances the framework’s dynamism, providing
greater flexibility and adaptability while improving computational efficiency and reducing power consumption. 


## Modeling of RC system using JJs for digit recognition task.

We chose a JTL to build the physical reservoir due to its
nonlinear characteristics, which allows it to function similarly
to an RNN. In our setup, the input layer consisted of voltage sources connected to random nodes of the JTL through input
resistors. When the current signal passed through these resistors into the JTL, quantized magnetic fluxes were generated
at the input nodes. These fluxes collided with one another,
causing the JJs to exhibit a wide range of voltage variations
depending on the input. The proposed circuit was designed to
be arranged in two dimensions on a substrate, allowing for an
extended range of propagation for the quantized fluxes. We
refer to this configuration as a crossbar JTL reservoir. The
reservoir was made up of 9 × 9 JJs. Voltage sources V1, V2,
V3, V4, and V5 were connected to the nodes at positions (4,
3), (7, 8), (3, 5), (1, 2), and (8, 0), respectively, where (i, j)
denotes the position of a node on the crossbar JTL.
We begin with the fundamental task of processing digitally generated images, specifically recognizing digits like the number ’2’ in Fig. 2. The input image consists of a 5 × 4 grid, totaling 20 pixels that are either black (0) or white (1). It is segmented into 5 rows, each containing 4 consecutive pixels,
and then converted into a pulse stream over 4 timeframes
and mapped as input into the reservoir’s JJ. Each timeframe
lasts 3 ms, where a write pulse (1.5 V, 1 ms) is applied for
white pixels, while black pixels receive no pulse (0 V).
Consequently, the spatial distribution of white pixels in the
digit ’2’ is encoded into temporal pulse patterns.
The goal is to extract the digit information, specifically
identifying the number ’2’, by analyzing these temporal features across the 5 pulse streams. This is accomplished using 5 JJs, each corresponding to a specific row of the image. The
reservoir state is determined by the collective resistance states
of these JJs. As the pulse streams are introduced, the reservoir
adapts to the input’s temporal structure, enabling the digit’s
recognition.
In this framework, the state of each JJ, after being activated,
encodes unique features corresponding to its specific row in
the input image. The overall reservoir state is determined by
the combined states of all the JJs, which can then be used for
pattern recognition tasks. With the trained readout function, the
system can identify digits, such as recognizing the number ’2’
from the input image.
The readout process is organized as a network comprising
5 input neurons and 10 output neurons. The reservoir state,
which is represented by the read currents from the 5 JJs, serves
as the input to this network. 
In the classification procedure, the output for each
of the 10 neurons is calculated by taking the dot
product between the 5 input values and the respective weight
vector for each neuron. The predicted digit is identified
by selecting the neuron that produces the highest dot productscore, and the associated label of that
neuron is chosen as the predicted digit. The readout function is trained using a supervised approach
with logistic regression applied to iteratively adjust the weights and
minimize classification error. The 10 digits, represented as 5 × 4 pixel grids, each have unique pixel patterns along the row direction,
which are mapped to 10 separate pulse streams for the JJs.
Reservoir states, which are determined by the combined resistance values of the 5 JJs
after processing these 10 images. The reservoir states for each digit are noticeably different, highlighting the system’s ability
to distinctly recognize the 10 digits. These reservoir states were then provided as input to the readout network during
both training and classification.

## Modeling of RC system using JJs for cart pole task.



