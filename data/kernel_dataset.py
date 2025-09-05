import itertools
from torch.utils.data import Dataset
import json
import os

dataset_types = ['train', 'validation', 'test', 'prompt','prompt_pcc', 'prompt_ml', 'prompt_math', 'train_pcc', 'validation_pcc', 'test_pcc', 'dataset_ml', 'dataset_math', 'prompt', 'dataset']

class KernelDataset(Dataset):
    def __init__(self, dataset_base, dataset_type='train', ignore_kernels=[]):
        if dataset_type not in dataset_types and dataset_type.split('_')[0] not in dataset_types:
            raise ValueError(f"Invalid dataset_type {dataset_type}. Choose from {dataset_types}, {dataset_type.split('_')[0]}.")
        
        self.dataset_base = dataset_base
        self.dataset_path = os.path.join(self.dataset_base, f'{dataset_type}.jsonl')
        self.ignore_kernels = ignore_kernels
        self.other_api_name = 'kernel_api' if 'pcc' in self.dataset_path.split('/')[-1] else 'parallel_api'

        self.dataset_base = os.path.abspath(dataset_base)
        self.dataset_path = os.path.join(self.dataset_base, f'{dataset_type}.jsonl')

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        self.ignore_kernels = ignore_kernels

        self.kernels = self._load_kernels()
        self.dataset = self._create_dataset()
        self.kernel_dict = self._create_kernel_dict()
        self.combinations = self._create_combinations()

    def _load_kernels(self):
        kernels = []

        with open(self.dataset_path, 'r') as f:
          for line in f:
              kernels.append(json.loads(line))

        return kernels

    def _create_dataset(self):
        kernals = {}
        for kernel in self.kernels:
            the_code = kernel['PCC']['gpt-4o-mini'].items() if 'pcc' in self.dataset_path.split('/')[-1] else kernel['code'].items()
            kernals[f"{kernel['kernel_name']}-{kernel[self.other_api_name]}"] = '\n'.join(f'{filename}:\n{code}' for filename, code in the_code)
        
        # kk = {f"{kernel['kernel_name']}-{kernel['parallel_api']}": '\n'.join(f'{filename}:\n{code}' for filename, code in kernel['code'].items()) for kernel in self.kernels}
        return kernals

    def _create_kernel_dict(self):
        kernel_dict = {}

        for kernel in self.kernels:
            name = kernel['kernel_name']
            api = kernel[self.other_api_name]
          
            if name not in kernel_dict:
                kernel_dict[name] = []

            if api not in kernel_dict[name]:
                kernel_dict[name].append(api)

        return kernel_dict

    def _create_combinations(self):
        combinations = []
        # try:
        #     existing_out = os.listdir(self.output_path)
        # except FileNotFoundError:
        #     os.mkdir(self.output_path)#, exist_ok=True)
        #     existing_out = os.listdir(self.output_path)
        for name, apis in self.kernel_dict.items():
            if len(apis) > 1:
                for api1, api2 in itertools.permutations(apis, 2):
                    if f'{name}_{api1}_{api2}' in self.ignore_kernels:
                        continue
                    combinations.append((name, api1, api2))
            else:
                api1 = apis[0]
                api2 = 'omp'
                if f'{name}_{api1}_{api2}' in self.ignore_kernels:
                    continue
                combinations.append((name, api1, api2))
                  
        return combinations

    def __getitem__(self, index):
        target_api = 'omp'

        if not self.combinations:
            raise IndexError("Dataset is empty, no combinations available")

        if isinstance(index, slice):
            # Handle slice indices
            start, stop, step = index.indices(len(self.combinations))
            return [
                (
                    name,
                    api1,
                    self.dataset.get(f'{name}-{api1}', "// No code available"),
                    api2,
                    self.dataset.get(f'{name}-{target_api}', "// No code available")
                )
                for name, api1, api2 in self.combinations[start:stop:step]
            ]
        else:
            # Handle integer index
            if index < 0 or index >= len(self.combinations):
                raise IndexError(f"Index {index} out of bounds for dataset with {len(self.combinations)} combinations")

            name, api1, api2 = self.combinations[index]
            code1 = self.dataset.get(f'{name}-{api1}', "// No code available")
            code2 = self.dataset.get(f'{name}-{target_api}', "// No code available")
            return name, api1, code1, target_api, code2

    def __len__(self):
        return len(self.combinations)
