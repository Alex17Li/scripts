import os
from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import imageids_to_dataset
import random
from sklearn.cluster import KMeans
from brtdevkit.data import Dataset
from email.mime import image
import itertools
from pathlib import Path
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset as torchDataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import yaml
from pathlib import Path 
from sklearn.decomposition import PCA
torch.cuda.empty_cache()

import imageio

class Datasetpreparer(torchDataset):    
        def __init__(self, data, preprocess=None, transform=None, device="cuda"):
            self.data = data
            self.transform =  transform 
            self.preprocess = preprocess
            self.device = device 
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path = self.data.loc[idx, 'image_path']
            try:
                img_pil = self.transform(Image.fromarray(imageio.imread(img_path)))
                # Download and open the image        
                img_pil = self.preprocess(img_pil).to(self.device)
            except Exception as e:
                print(e)
                img_path = self.data.loc[0, 'image_path']
                img_pil = self.transform(Image.fromarray(imageio.imread(img_path)))
                img_pil = self.preprocess(img_pil).to(self.device)
                return img_pil, img_path
            return img_pil, img_path

class ImageSimilarity: 
    def __init__(self, images_full_path, data_base_path, dataset_name='image_list', overwrite=False):
        self.data_base_path = data_base_path
        os.makedirs(self.data_base_path, exist_ok=True)
        self.images_base_path = data_base_path + "/images"
        self.images_full_path = images_full_path
        
        self.image_path_df = None 
        self.transform = T.Compose([
            T.Resize((224,224))])
        
        self.inference_set = None 
        self.embeddings_np = None
        self.image_paths = None 
        self.image_paths_df = None
        self.embeddings_save_name = f"{dataset_name}_embeddings.npz"
        self.embeddings_image_paths = Path(self.data_base_path) / f"{dataset_name}_embeddings_image_paths.csv"
        self.embeddings_save_loc = Path(self.data_base_path) / self.embeddings_save_name
        self.overwrite = overwrite 
        self.model = None 
        self.preprocessor = None 
        self.sorted_scores = None 
        self.sorted_scores_save_loc = data_base_path 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        
    def prepare_images_path_df(self): 
        self.image_path_df = pd.DataFrame(data=self.images_full_path, columns=['image_path'])
        return self.image_path_df
    
    def prepare_dataloader(self): 
        self.inference_set = Datasetpreparer(data=self.image_path_df, preprocess=self.preprocessor, transform=self.transform, device=self.device)
        inference_loader = DataLoader(self.inference_set, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
        self.total = len(self.image_path_df) / 64
        return inference_loader
 
    def load_model(self): 
        self.model, self.preprocessor = clip.load("ViT-B/32", device=self.device)
        return self.model, self.preprocessor

    def build_embeddings(self): 
        outputs = []
        image_paths = []
        if (os.path.isfile(self.embeddings_save_loc)==True) and (self.overwrite==False): 
            self.embeddings_np, self.image_paths = self.load_embeddings()
            return self.embeddings_np, self.image_paths 
        else:
            inference_loader = self.prepare_dataloader()
            for idx, batch in tqdm(enumerate(inference_loader), total=self.total): 
                batch, paths = batch[0], batch[1]
                with torch.no_grad():
                        batch = batch.to(self.device)
                        image_features = self.model.encode_image(batch)
                        outputs.append(image_features)  
                        image_paths.extend(paths)
            outputs = torch.cat(outputs, dim=0)
            self.image_paths = image_paths 
            self.embeddings_np = outputs.detach().cpu().numpy()
            return self.embeddings_np, self.image_paths
            
    def save_embeddings(self): 
            np.savez(self.embeddings_save_loc, self.embeddings_np) 
            image_paths_df = pd.DataFrame(data=self.image_paths, columns=['image_path'])
            image_paths_df.to_csv(self.embeddings_image_paths, index=False)
            return None 
        
    def load_embeddings(self): 
        embeddings_save_loc = str(self.embeddings_save_loc)
        embeddings = np.load(embeddings_save_loc)
        self.embeddings_np = embeddings[embeddings.files[0]]       
        self.image_paths_df= pd.read_csv(self.embeddings_image_paths )
        return self.embeddings_np, self.image_paths_df 

    def get_image_embedding(self, image_path): 
        transformed_image = self.transform(Image.open(image_path))
        img = self.preprocessor(transformed_image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(img)
        embeddings_np = embedding.detach().cpu().numpy()
        return embeddings_np
    
    def get_similar_images(self, image_path, calcuate_embeddings=False, images_base_path=None): 
        if images_base_path==None:
            images_base_path = self.images_base_path
        full_image_path = os.path.join(images_base_path, image_path)
        embeddings_np, image_paths_df = self.load_embeddings()
        image_paths = image_paths_df['image_path'].tolist()
        if calcuate_embeddings==True:
            reference_embedding = self.get_image_embedding(full_image_path)
        else :
            embedding_index = image_paths_df[image_paths_df['image_path'] == full_image_path].index[0]
            reference_embedding = embeddings_np[embedding_index, :]
        
        similarity_score = np.dot(reference_embedding, embeddings_np.T)/(np.linalg.norm(reference_embedding)*np.linalg.norm( embeddings_np, axis=1))
        score_df = pd.DataFrame(data=list(similarity_score.T), columns=['similarity_score'])
        score_df['image_path'] = image_paths 
        self.sorted_scores = score_df.sort_values(by='similarity_score', ascending=False)
        image_name = image_path.split("/")[-1].replace(".png","")
        save_loc_sorted_scores = Path(self.sorted_scores_save_loc) / f"{image_name}.csv"
        self.sorted_scores.to_csv(save_loc_sorted_scores, index=False)
        return self.sorted_scores
    
    def similar_images_with_text(self, text_prompt):
        
        embeddings_np, image_paths_df = self.load_embeddings()
        image_paths = image_paths_df['image_path'].tolist()
        text = clip.tokenize(text_prompt).to(self.device)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.detach().cpu().numpy()
        reference_embedding = text_features
            
        similarity_score = np.dot(reference_embedding, embeddings_np.T)/(np.linalg.norm(reference_embedding)*np.linalg.norm( embeddings_np, axis=1))
        score_df = pd.DataFrame(data=list(similarity_score.T), columns=['similarity_score'])
        score_df['image_path'] = image_paths 
        self.sorted_scores = score_df.sort_values(by='similarity_score', ascending=False)
        image_name = text_prompt.replace(" ","_")
        save_loc_sorted_scores = Path(self.sorted_scores_save_loc) / f"{image_name}.csv"
        self.sorted_scores.to_csv(save_loc_sorted_scores, index=False)
        return self.sorted_scores
        
        
    def plot_image_grid(self, image_paths_df, tail=0  ,nrows=5, ncols=2):
        
        # Assume image_paths is your list of image file paths
        if tail > 0:
            image_paths_df = image_paths_df.tail(tail)
        image_paths = image_paths_df['image_path'].tolist()
        n_rows = min(nrows, int(image_paths_df.shape[0] // 2 ))
        fig, axes = plt.subplots(n_rows, ncols, figsize=(10, nrows*5))
        for i, ax in enumerate(axes.flatten()):
            if i < len(image_paths)-1:
                image_path = image_paths[i]
                title = image_path.split("/")[-1]
                img = Image.open(image_path)
                ax.imshow(img)
                ax.set_title(f'{title}  indx: {i}', fontsize=6)
            ax.axis('off')

        plt.tight_layout()
        plt.show()


    def extract_embeddings(self): 
        self.prepare_images_path_df()
        self.prepare_dataloader()
        self.build_embeddings()
        self.save_embeddings()
        return None
    
def get_images(save_dir, df):
    save_dir = Path(save_dir)
    dirs = [save_dir / d for d in df.id]
    image_paths = []
    for i, dir in tqdm(enumerate(dirs)):
        if i % 1500 == 0:
            print(100 * i / len(dirs))
        for fname in os.listdir(dir):
            if 'debayeredrgb' in fname and fname.endswith('.png'):
                image_paths.append(str(object=dir / fname))
    return image_paths

def diversify_dataset(dsetname:str, n_images_final: int, kind: str):
    aletheia_ds = Dataset.retrieve(name=dsetname)
    aletheia_df = aletheia_ds.to_dataframe()
    dataset_save_dir = os.environ['DATASET_PATH'] + "/" + dsetname
    save_file = Path(dataset_save_dir) / "image_ids.npy"
    if not os.path.exists(save_file):
        print("Downloading images")
        os.makedirs(name=dataset_save_dir, exist_ok=True)
        aletheia_ds.download(dataset_save_dir)
        print("Looking through directory for images")
        image_paths = list(get_images(save_dir=dataset_save_dir + '/images', df=aletheia_df))
        np.save(save_file, image_paths)
    else:
        image_paths = np.load(save_file).tolist()
    image_paths = [p for p in image_paths if p.endswith('.png')]
    print(f"Looking at similarity for {len(image_paths)} images")
    sim = ImageSimilarity(images_full_path=image_paths, data_base_path=dataset_save_dir, dataset_name=dsetname, overwrite=False)
    sim.extract_embeddings()
    embeddings_np, paths_df = sim.load_embeddings()
    print("Running Kmeans")
    kmeans = KMeans(n_clusters=n_images_final, random_state=0, n_init="auto")
    kmeans.fit(embeddings_np)
    final_paths = [None for _ in range(n_images_final)]

    print("Choosing one random image from each cluster")
    order = list(enumerate(kmeans.labels_))
    random.shuffle(order)
    for i, l in order:
        if final_paths[l] == None:
            final_paths[l] = paths_df.image_path.iloc[i]
    imids = [p.split('_')[-1][:-4] for p in final_paths]
    fin_df = aletheia_df[aletheia_df.id.isin(imids)]
    print(len(fin_df))
    print(len(imids))
    score = kmeans.inertia_ / len(aletheia_df)
    print(f"KMEANS SCORE: {score}")

    desc = f"{aletheia_ds['description']} Select diverse to get {len(imids)} images, diversity {score}"

    from aletheia_dataset_creator.dataset_tools.aletheia_dataset_helpers import imageids_to_dataset
    imageids_to_dataset(image_ids=imids, dataset_name=f"{dsetname}_diverse_{n_images_final}", dataset_description=desc, dataset_kind=kind, production_dataset=False)

if __name__ == "__main__":
    diversify_dataset("mannequin_in_dust_v0", 1500, Dataset.KIND_ANNOTATION)
    # diversify_dataset("dynamic_manny_in_dust_raw", 4000, Dataset.KIND_IMAGE)
