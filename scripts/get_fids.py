import os
import os.path as osp
import numpy as np
import csv

import sys
sys.path.append('../')
sys.path.append('../utils/')
sys.path.append('../DefectGeneration/')

from utils.fid import fid_calculator
from utils.prdc import compute_prdc


def get_fids(base_path, sources):
    fids = {}
    for source1, source2 in sources:
        print(source1, source2)
        source1_paths = []
        source2_paths = []
        for dirpath, dirnames, filenames in os.walk(base_path):
            dirpath = dirpath.replace('\\', '/')
            if source1 in dirpath and '/images' in dirpath and 'train' in dirpath:
                if "Good" in source2 and 'Good' in dirpath:
                    print('skipping', dirpath)
                else:
                    source1_paths.append(dirpath)
            if source2 in dirpath and 'images' in dirpath and 'train' in dirpath:
                source2_paths.append(dirpath)
        
        print(source1_paths[:3])
        
        print(source2_paths[:3])
        ret_prdc = compute_prdc(source1_paths, source2_paths, use_multiprocessing=False, batch_size=2, nearest_k=5, max_images=5000, total_max_images=50000)
        result = {}
        print(ret_prdc)
        for key, value in ret_prdc.items():
            result[key] = value
        # ret_fid = fid_calculator(source1_paths, source2_paths, 32, max_images=5000, total_max_images=50000)
        # result['fid'] = ret_fid

    return result

    

def main():
    base_path = '/nn-seidenader-gentofte/TJSD/VisData/'
    # sources = [('Synthetic/CAM2/', 'Real/CAM2/'), ('Real/CAM2/', 'Real_generative_5_256_4_19'), ('Real/CAM2/', 'Real/CAM2/Good/'), ]

    # sources = [('DiffusionSlerpImg2Img/CAM2/', 'Real/CAM2/'), ('DiffusionSlerpSampler/CAM2/', 'Real/CAM2/'), ('synth2real/CAM2/', 'Real/CAM2/'), ('Synthetic/CAM2/', 'Real/CAM2/')]
    # sources = [('DiffusionSlerpImg2Img/CAM2/', 'Real/CAM2/')]
    
    source1 = sys.argv[1]
    source2 = sys.argv[2]
    print(sys.argv)
    print(source1, source2)
    # with open('fid_out.csv', 'a') as output_file:
    #     writer = csv.writer(output_file)
    #     writer.writerow(['source1', 'source2', 'precision', 'recall', 'density', 'coverage', 'fid'])

    # return
    input_sources = [(source1, source2)]
 
    fids = get_fids(base_path, input_sources )
    
    print(fids)
    #Append the fid result to the csv file on a new line
    with open('fid_out.csv', 'a') as output_file:
        writer = csv.writer(output_file)
        # limit the precision for all metrics to 4 decimal places
        precision = round(fids['precision'], 4)
        recall = round(fids['recall'], 4)
        density = round(fids['density'], 4)
        coverage = round(fids['coverage'], 4)
        fid = round(fids['fid'], 4)
        writer.writerow([source1, source2, precision, recall, density, coverage, fid])    



    # writer.writerow([source1, source2, fids['precision'], fids['recall'], fids['density'], fids['coverage'], fids['fid']])

    # [{'precision': 0.018524773311855335, 'recall': 0.00010203561042803938, 'density': 0.0037517212710124976, 'coverage': 0.00020407122085607876}
    # , {'precision': 0.09052631578947369, 'recall': 0.08423185845306451, 'density': 0.02571578947368421, 'coverage': 0.0287874457637247}
    # , {'precision': 0.5688062387522496, 'recall': 0.05910708988430733, 'density': 0.38346730653869227, 'coverage': 0.1805843963215663}]
if __name__ == "__main__":
    main()
