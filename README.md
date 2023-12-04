*Advancing Spatial-Temporal Rock Fracture Prediction with Virtual Camera-Based Data
Augmentation*

## Abstract
Predicting rock fractures in unexcavated areas is a critical yet challenging aspect of
geotechnical projects. This task involves forecasting the fracture mapping sequences
for unexcavated rock faces using the sequences from excavated ones, which is well-suited for spatial-temporal deep learning techniques. Fracture mapping sequences for
deep learning model training can be achieved based on field photography. However,
the main obstacle lies in the insufficient availability of high-quality photos. Existing data
augmentation techniques rely on slices taken from Discrete Fracture Network (DFN)
models. However, slices differ significantly from actual photos taken from the field. To
overcome this limitation, this study introduces a new framework that uses Virtual
Camera Technology (VCT) to generate “virtual photos” from DFN models. The external
(e.g., camera location, direction) and internal parameters (e.g., focal length, resolution,
sensor size) of cameras can be considered in this method. The “virtual photos”
generated from the VCT and conventional slicing method have been extensively
compared. The framework is designed to adapt to any distribution of field fractures and
camera settings, serving as a universal tool for practical applications. The whole
framework has been packaged as an open-source tool for rock “photos” generation. An
open-source benchmark database has also been established based on this tool. To
validate the framework's feasibility, the Predictive Recurrent Neural Network
(PredRNN) method is applied to the generated database. A high degree of similarity is
observed between the predicted mapping sequences and the ground truth. The model
successfully captured the dynamic changes in fracture patterns across different
sections, thereby confirming the framework's practical utility.

## Data Demo

![FixStep Animation](./images/FixStep05_PBSet1_20_80.gif)

*Figure 1: FixStep Mapping with color-image*

![FixStep Animation 2](./images/FixStep05_PBSet1_20_80_2.gif)

*Figure 2: FixStep Mapping with binary-image*

![NRandStep Animation](./images/NRandStep_PBSet1_20_80.gif)

*Figure 3: RandStep Mapping with color-image*

![NRandStep Animation 2](./images/NRandStep_PBSet1_20_80_2.gif)

*Figure 4: RandStep Mapping with binary-image*
## Usage

If you are using a sequence forecasting model like PredRNN for rock fracture mapping prediction, you may need a substantial amount of data. Here, we provide a tool to synthesize all the data you need!


## full code and data set

coming soon