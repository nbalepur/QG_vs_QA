import pickle
import os

class Checkpoint:

    def __init__(args):

        self.partition_map = {'full': (0, len(input_prompts)),
        'first_half': (0, int(0.5 * len(input_prompts))),
        'second_half': (int(0.5 * len(input_prompts)), len(input_prompts)),
        'first_quarter': (0, int(0.25 * len(input_prompts))),
        'second_quarter': (int(0.25 * len(input_prompts)), int(0.5 * len(input_prompts))),
        'third_quarter': (int(0.5 * len(input_prompts)), int(0.75 * len(input_prompts))),
        'fourth_quarter': (int(0.75 * len(input_prompts)), len(input_prompts)),
        'first_eighth': (0, int(0.125 * len(input_prompts))),
        'second_eighth': (int(0.125 * len(input_prompts)), int(2*0.125 * len(input_prompts))),
        'third_eighth': (int(2*0.125 * len(input_prompts)), int(3*0.125 * len(input_prompts))),
        'fourth_eighth': (int(3*0.125 * len(input_prompts)), int(4*0.125 * len(input_prompts))),
        'fifth_eighth': (int(4*0.125 * len(input_prompts)), int(5*0.125 * len(input_prompts))),
        'sixth_eighth': (int(5*0.125 * len(input_prompts)), int(6*0.125 * len(input_prompts))),
        'seventh_eighth': (int(6*0.125 * len(input_prompts)), int(7*0.125 * len(input_prompts))),
        'eighth_eighth': (int(7*0.125 * len(input_prompts)), len(input_prompts)),
        }

        if args.partition not in partition_map:
            raise ValueError(f"The given partition is invalid: {args.partition}")
        start, end = partition_map[partition]
        self.start = start
        self.end = end

        self.results_dir = f'{args.results_dir}/{args.model_nickname}
        self.num_shots = args.num_shots
        self.partition = args.partition

    def set_directories(pt):

        if args.partition == 'full':
            final_res_dir = f'{self.results_dir}/{pt.value}_{self.num_shots}_shot.pkl'
            final_res_dir_temp = f'{self.results_dir}/{pt.value}_{self.num_shots}_shot_temp.pkl'
        else:
            final_res_dir = f'{self.results_dir}/{pt.value}_{self.num_shots}_shot_{self.partition}.pkl'
            final_res_dir_temp = f'{self.results_dir}/{pt.value}_{self.num_shots}_shot_{self.partition}_temp.pkl'

        self.final_res_dir = final_res_dir
        self.final_res_dir_temp = final_res_dir_temp

    def load_checkpoint():

        if os.path.exists(self.final_res_dir):
            with open(self.final_res_dir, 'rb') as handle:
                outputs = pickle.load(handle)
                return outputs, self.start + len(outputs['raw_text'])

        if not os.path.exists(self.final_res_dir_temp):
            return {'raw_text': [], 'prompt': []}, self.start
        
        with open(self.final_res_dir_temp, 'rb') as handle:
            outputs = pickle.load(handle)

        return outputs

    def save_checkpoint(is_final, outputs):

        out_dir = self.final_res_dir if is_final else self.final_res_dir_temp

        folder_path = '/'.join(out_dir.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        with open(out_dir, 'wb') as handle:
            pickle.dump(answers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_bounds():
        return self.start, self.end

    def get_final_dir():
        return self.final_res_dir