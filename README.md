# **MPAP**

![image](https://github.com/Hanxiaoxiao123/MSFragTox/blob/main/f1.jpg) #待修改github路径

# **Training of MPAP**

We collected the in vitro adsorption coefficient data (Kd) from 91 published studies and systematic literature searches (2021–2023), covering 1,101 adsorption records across 403 organic pollutants and six microplastic types (PE, PS, PP, PVC, PET, PA) in freshwater, seawater, and ultrapure water. Then we integrated molecular fingerprints (ECFPs for pollutants and PolyBERT for MPs), graph-based structural features, and environmental parameters (MP particle size, water environment) to form multimodal representations. The molecular fingerprints were generated via ECFPs and PolyBERT, while graph features were extracted from atomic/bond descriptors. Environmental covariates (particle size as continuous variable; water type as one-hot vector) were concatenated to structural features.

To train the multimodal Siamese neural network (MPAP), data division stratified by MP type and water environment, Bayesian hyperparameter optimization (lr, batch size, L2 regularization, dropout rate), and training with Adam optimizer were conducted. The model fused pollutant/MP features via attention-guided fusion blocks, integrated environmental factors, and predicted log10Kd through three fully connected layers. The code for model architecture, training, and the open-access web tool is available at https://github.com/xx and http://mpap.envwind.site:8004/.  #待修改github路径


# **How can our models predict kd of MPs and pollutants?**

## **Requirements:**
conda env create -f environment.yaml

## **Step 1: Convert  text to npy input file using MPAP predata.**

Create a txt text containing the title line ‘category psmiles compound smiles average size water3 logkd poly_smiles’.
And the corresponding values, catagory is the type of microplastic, psmiles is the corresponding smile of the microplastic, compound is the name of the pollutant, smiles is the corresponding smile, average size is the particle size of the microplastic, measured in micrometers. Water3 is the type corresponding to the water environment, where 1 is freshwater, 2 is ultrapure water, and 3 is seawater. Logkd is the actual kd value, which can be input as 1 in the prediction model without affecting the prediction results. Poly_smiles are the corresponding polymers for microplastics
Modify the content of line 889 in predata.py, enter the txt file name just created, and the corresponding output npy file package.Finally run it.


## **Step 2: predict kd**   

Modify the file path in predication.py Line 74, Line 781, modify the path of the training dataset in Line 176, modify Line 177 to the path of the npy file package just output in the first step, and modify the prediction result path output in Line 934.Finally run it.



# **License**  

This project is licensed under the MIT License - see the LICENSE.md file for details.


