import glob
class DatasetSTS:
    def __init__(self, gt_folder):
        self.root = gt_folder
        self.gt_files = self.get_gt_files()
        self.input_files = self.get_inp_files()
        
    def get_gt_files(self):
        all_files = glob.glob(self.root+"/STS2016.gs.*.txt")
        return list(all_files)
    
    def get_inp_files(self):
        all_files = glob.glob(self.root+"/STS2016.input.*.txt")
        return list(all_files)
        
    def pre_process(self):
        data = []
        for g_file, inp_file in zip(self.gt_files, self.input_files):
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
        self.data = data
    
    def get_sentence_pairs(self):
        sent1 = [x for x,_,_ in self.data]
        sent2 = [x for _,x,_ in self.data]
        return sent1, sent2
    
    def get_gt_scores(self):
        scores = [x for _,_,x in self.data]
        return scores

    