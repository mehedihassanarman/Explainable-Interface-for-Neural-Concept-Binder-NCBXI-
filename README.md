# Explainable Interface for Neural Concept Binder (NCBXI)

Unsupervised learning is one of the most challenging problems in machine learning, particularly when applied to complex visual data. Unlike supervised learning, which requires labeled datasets, unsupervised methods must extract meaningful patterns and representations from raw data without explicit guidance. This lack of supervision makes it difficult to ensure that learned representations are both useful and interpretable. Traditional unsupervised learning methods, such as contrastive learning, often struggle to capture human-understandable concepts in visual data. The Neural Concept Binder (NCB) addresses this by binding learned features to high-level, human-understandable concepts. This approach allows the model to automatically discover relationships between visual elements and organize them into structured concepts, offering a significant leap forward in both the interpretability and applicability of unsupervised learning. 

![home.png](https://github.com/mehedihassanarman/Explainable-Interface-for-Neural-Concept-Binder-NCBXI-/blob/main/static/images/icons/home.png)

The NCB’s technical innovation lies in how it structures the learning process. Unlike typical unsupervised learning methods, which may produce abstract or fragmented representations, NCB organizes the learned features in a way that aligns with human-understandable concepts. The model operates by binding learned visual representations to a latent space that is interpretable in terms of concepts such as color, shape, and object type. This structured approach enables the model to not only recognize features in isolation but also understand their relationships, allowing for more coherent and robust learning. Furthermore, the NCB framework introduces a concept-based bottleneck mechanism that helps the model focus on the most meaningful and relevant aspects of the data. This makes NCB a powerful tool for tasks such as zero-shot learning, concept-based reasoning, and image generation, all without the need for labeled training data.

Our motivation for developing a web application around the NCB is to bring this complex and innovative model to life in a way that is engaging and accessible. By visualizing the implicit, comparative, and interventional processes of NCB, we provide a hands-on way for users to interact with and understand the power of concept-based learning. The web app allows users to explore how NCB binds features to concepts, compare it with traditional unsupervised learning methods, and observe the effects of various interventions on the model’s learned representations. This interactive approach demystifies the inner workings of NCB, making it easier to appreciate its contributions to the field of unsupervised learning and its potential applications. By offering a dynamic and visual representation of such a sophisticated model, we hope to inspire further research and experimentation in structured, unsupervised learning.


## Installtion
To run this project, please follow the steps below:

1. Create a folder named `data` and download the folder `CLEVR-4-1` from the following link [CLEVR-4-1.zip](https://hessenbox.tu-darmstadt.de/getlink/fiVCLMaZkEuf5f6HYG58sshV/CLEVR-4-1.zip). After extracting, ensure the path structure is as follows: `data/CLEVR-4-1/test/images/CLEVR_4_classid_0_000000.png` 

2. Create a folder named `model` and download the folder `CLEVR-4` from the following link [CLEVR-4.zip](https://hessenbox.tu-darmstadt.de/getlink/fi6WzuWtQ87Px5P3ewEVNQyZ/CLEVR-4.zip). After extracting, ensure the path structure is as follows: `model/CLEVR-4/retbind_seed_2/best_model.pt"`   

3. Open root folder and execute the following commands in the terminals:

4. `git clone https://github.com/mehedihassanarman/Explainable-Interface-for-Neural-Concept-Binder-NCBXI-.git`

5. `cd Explainable-Interface-for-Neural-Concept-Binder-NCBXI-`

6.  ```bash
python -m venv .venv
.venv/Scripts/activate

7. `pip install -r requirements.txt`
   
8. `cd NeuralConceptBinder`
   
9. `pip install -e sysbinder`

10. `cd ..`
    `python app.py`


The required Python version for this project is 3.10.10.
