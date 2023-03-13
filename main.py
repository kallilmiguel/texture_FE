import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction for texture database with classification")
    #data, paths, and other settings of general setup
    parser.add_argument('--dataset', type=str, default='DTD', help='Variation of wood dataset to be used, other options:1000:500; 500:500 or 500:500:OGRN')
    parser.add_argument('--output_path', type=str, default= '/home/kallilzie/results/', help='Path for saving each run')    
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID or list of ids ID1,ID2,... (see nvidia-smi), wrong ID will make the script use the CPU')
    # Feature extraction
    parser.add_argument('--backbone', type=str, default='resnet18', help='Name of an architecture to experiment with (see models.py')
    parser.add_argument('--input_dimm', type=int, default=224, help='Image input size (single value, square). The standard is a forced resize to 224 (square)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size, increase for better speed, if you have enough VRAM')

    # Classification procedure
    parser.add_argument('--repetitions', type=int, default=10, help='Number of repetitions for classification')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross validation')
    parser.add_argument('--base_seed', type=int, default=666999, help='Deterministic seed for the execution')

    return parser.parse_args()
    

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import datasets as torchDatasets
from torchvision import transforms
import models
import datasets as ds
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import numpy as np

import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.preprocessing import StandardScaler,PowerTransformer
import timm
import pickle
import random
from joblib import parallel_backend

DATASETS_ = {'DTD' : ds.DTD,
                 'FMD' : ds.FMD,
                 'USPtex': ds.USPtex,
                 'LeavesTex1200': ds.LeavesTex1200,
                 'KTH-TIPS2-b': ds.KTH_TIPS2_b,
                 'Outex' : ds.Outex,
                 'MINC': ds.MINC,
                 'GTOS': ds.GTOS,
                 'GTOS-Mobile': ds.GTOS_mobile
                }

if __name__ == "__main__":
    
    args = parse_args()
   
    features_path = f'{args.output_path}/{args.dataset}/feature_matrix/'
    
    results_path = f'{args.output_path}/{args.dataset}/classification/'
    
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    features_file = features_path+f'{args.backbone}_K={args.folds}_size={args.input_dimm}.pkl'

    results_file = results_path + f'{args.backbone}_K={args.folds}_size={args.input_dimm}.pkl'

    if not os.path.isfile(results_file):
        print("Results not yet calculated, veryfing features file...")

        if not os.path.isfile(features_file):
            
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{args.gpu}')
            else:
                device = torch.device('cpu')

            averages = (0.485, 0.456, 0.406)
            variances = (0.229, 0.224, 0.225)
            _transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(averages, variances)
            ])  
                
            if args.input_dimm != None:
                _transform.transforms.append(transforms.Resize((args.input_dimm, args.input_dimm)))
            
            dataset = DATASETS_[args.dataset](root=ds.DATASETS_PATH[args.dataset],transform=_transform)
                    
            model = models.timm_pretrained_features(args.backbone).to(device)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

            feature_size = model.get_features(torch.ones((1,3, 224, 224)).to(device)).cpu().detach().numpy().shape[1]
            print("Features not yet calculated, extracting", feature_size, "descriptors...")
            X = np.empty((0, feature_size))
            y = np.empty((0))
            for batch in dataloader:            
                image_batch, label_batch = batch[0].to(device), batch[1]
                
                X = np.vstack((X, model.get_features(image_batch).cpu().detach().numpy()))
                y = np.hstack((y, label_batch))
            
            with open(features_file, 'wb') as f:
                    pickle.dump([X,y], f)
                    
        else:
            print("Loading already calculated features...")
            
            with open(features_file, "rb") as f:
                X, y = pickle.load(f)
            
        print("Now classifying...")
        
        kfold = KFold(n_splits=args.folds, shuffle=True)
        
        gtruth_= []
        preds_SVM, preds_KNN, preds_LDA, preds_RF = [], [], [], []
        accs_SVM, accs_KNN, accs_LDA, accs_RF = [], [], [], []
        print("Repetition: ")
        for i in range(args.repetitions):
            print(i+1)
            seed = args.base_seed*(i+1)
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
            for train_index, test_index in kfold.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
                std = StandardScaler()
                X_train = std.fit_transform(X_train)
                X_test = std.transform(X_test)
                
                gtruth_.append(y_test)
                with parallel_backend('threading', n_jobs=16):
                    KNN = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', 
                                                        leaf_size=30, p=2, metric='minkowski', metric_params=None)
                    
                    KNN.fit(X_train,y_train)
                    preds=KNN.predict(X_test)              
                    preds_KNN.append(preds)            
                    acc= sklearn.metrics.accuracy_score(y_test, preds)        
                    accs_KNN.append(acc*100) 
                    
                    LDA= LinearDiscriminantAnalysis(solver='lsqr', 
                                                            shrinkage='auto', priors=None,
                                                            n_components=None, 
                                                            store_covariance=False, 
                                                            tol=0.0001, covariance_estimator=None)
                    
                    LDA.fit(X_train,y_train)
                    preds=LDA.predict(X_test)              
                    preds_LDA.append(preds)            
                    acc= sklearn.metrics.accuracy_score(y_test, preds)
                    accs_LDA.append(acc*100)  
                    
                    svm = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', 
                                        coef0=0.0, shrinking=True, probability=False, tol=0.001,
                                        cache_size=200, class_weight=None, verbose=False, 
                                        max_iter=100000, decision_function_shape='ovr', 
                                        break_ties=False, random_state=seed)
                    
                    
                    svm.fit(X_train,y_train)
                    preds=svm.predict(X_test)            
                    preds_SVM.append(preds)            
                    acc= sklearn.metrics.accuracy_score(y_test, preds)
                    accs_SVM.append(acc*100)
                    
                
                
                
                results = {'gtruth_':gtruth_,
                        'preds_KNN':preds_KNN,
                        'accs_KNN':accs_KNN,
                        'preds_LDA':preds_LDA,
                        'accs_LDA':accs_LDA,
                        'preds_SVM':preds_SVM,
                        'accs_SVM':accs_SVM
                        }  
                
                with open(results_file, 'wb') as f:
                    pickle.dump(results, f)
                
        
    else:
        print("Already classified, loading results...")
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
            
    if args.repetitions > 1: #in this case avg acc is computed over iterations, and std over these averages
        knn, lda, svm, rf = [], [], [], []
        for it_ in range(args.repetitions):
            knn.append(np.mean(results['accs_KNN'][it_*args.folds: it_*args.folds + args.folds]))
            lda.append(np.mean(results['accs_LDA'][it_*args.folds: it_*args.folds + args.folds]))
            svm.append(np.mean(results['accs_SVM'][it_*args.folds: it_*args.folds + args.folds]))     
        results['accs_KNN'] = knn
        results['accs_LDA'] = lda
        results['accs_SVM'] = svm
        
            
        print('Acc: ', sep=' ', end='', flush=True)   
        print('KNN:', f"{np.round(np.mean(results['accs_KNN']), 1):.1f} (+-{np.round(np.std(results['accs_KNN']), 1):.1f})", sep=' ', end='', flush=True)      
        print(' || LDA:', f"{np.round(np.mean(results['accs_LDA']), 1):.1f} (+-{np.round(np.std(results['accs_LDA']), 1):.1f})", sep=' ', end='', flush=True)      
        print(' || SVM:', f"{np.round(np.mean(results['accs_SVM']), 1):.1f} (+-{np.round(np.std(results['accs_SVM']), 1):.1f})", sep=' ', flush=True)
