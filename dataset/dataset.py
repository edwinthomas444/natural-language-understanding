import glob
import os
import zipfile
import wget

class DatasetSTS:
    def __init__(self, download_url):
        # create raw folder
        self.download_url = download_url
        self.dataset_root = os.path.join(os.getcwd(),'dataset','STSRawData','sts2016-english-with-gs-v1.0')
        self.download_dataset()
        self.gt_files = self.get_gt_files()
        self.input_files = self.get_inp_files()
        self.pre_process()

    def download_dataset(self):
        if not self.download_url:
            return

        if not os.path.exists(os.path.dirname(self.dataset_root)):
            os.makedirs(os.path.dirname(self.dataset_root),exist_ok=True)

        zip_path = os.path.join(os.path.dirname(self.dataset_root),os.path.basename(self.download_url))
        wget.download(self.download_url,out=os.path.dirname(self.dataset_root))
        # extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(self.dataset_root))
    
    def get_gt_files(self):
        all_files = glob.glob(os.path.join(self.dataset_root,"STS2016.gs.")+"*.txt")
        return sorted(list(all_files))
    
    def get_inp_files(self):
        all_files = glob.glob(os.path.join(self.dataset_root,"STS2016.input.")+"*.txt")
        return sorted(list(all_files))
        
    def pre_process(self):
        data = []
        # store pre-processed data per file
        file_data = {}
        for g_file, inp_file in zip(self.gt_files, self.input_files):
            dataset_type = g_file.split('.')[-2]
            with open(g_file,"r",encoding='utf-8') as fg:
                lines_fg = fg.readlines()
                valid_pos = [i for i,x in enumerate(lines_fg) if x!="\n"]
                valid_gt = [lines_fg[i] for i in valid_pos]
                with open(inp_file,"r",encoding='utf-8') as fi:
                    lines_fi = fi.readlines()
                    valid_lines = [lines_fi[i] for i in valid_pos]
            # process valid lines
            for vline, vgt in zip(valid_lines, valid_gt):
                f_line, s_line = vline.split("\t")[:2]
                entry = [f_line, s_line, vgt.strip("\n")]
                data.append(entry)

                file_data.setdefault(dataset_type,[]).append(entry)
        file_data['all'] = data
        self.data = data
        self.file_data = file_data
    
    def get_sentence_pairs(self, data):
        sent1 = [x for x,_,_ in data]
        sent2 = [x for _,x,_ in data]
        return sent1, sent2
    
    def get_gt_scores(self, data):
        scores = [x for _,_,x in data]
        return scores

class DatasetSQUAD:
    def __init__(self, download_url):
        # create raw folder
        self.download_url = download_url
        self.dataset_root = os.path.join(os.getcwd(),'dataset','SQUADRawData')
        self.download_dataset()
        self.train_file = os.path.join(self.dataset_root,'train-v2.0.json')
        self.test_file = os.path.join(self.dataset_root,'dev-v2.0.json')

    def download_dataset(self):
        if not self.download_url:
            return

        train_url = self.download_url['train']
        test_url = self.download_url['test']
        if not os.path.exists(self.dataset_root):
            os.makedirs(self.dataset_root, exist_ok=True)

        wget.download(train_url,out=self.dataset_root)
        wget.download(test_url,out=self.dataset_root)

    