import itertools
from torch.utils.data import Dataset
import json
import os

dataset_types = ['train', 'validation', 'test', 'prompt', 'temp']

class KernelDataset(Dataset):
    def __init__(self, dataset_base, dataset_type='train', ignore_kernels=[], parallel_apis=['cuda', 'omp'], from_api=None, to_api=None):
        if dataset_type not in dataset_types:
            raise ValueError(f"Invalid dataset_type. Choose from {dataset_types}.")

        self.dataset_base = os.path.abspath(dataset_base)
        self.dataset_path = os.path.join(self.dataset_base, f'{dataset_type}.jsonl')

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        self.ignore_kernels = ignore_kernels
        self.from_api = from_api
        self.to_api = to_api

        self.parallel_apis = parallel_apis
        self.kernels = self._load_kernels()
        self.dataset = self._create_dataset()
        self.kernel_dict = self._create_kernel_dict()
        self.combinations = self._create_combinations()
        self.filter_kernels_with_mismatched_files()

    def _load_kernels(self):
        kernels = []

        with open(self.dataset_path, 'r') as f:
          for line in f:
              kernel = json.loads(line)
              if kernel['parallel_api'] not in self.parallel_apis:
                  continue
              kernels.append(kernel)

        return kernels

    def _create_dataset(self):
        return {
            f"{kernel['kernel_name']}-{kernel['parallel_api']}": kernel['code']
            for kernel in self.kernels
        }
    def _create_kernel_dict(self):
        kernel_dict = {}

        for kernel in self.kernels:
            name = kernel['kernel_name']
            api = kernel['parallel_api']

            if name not in kernel_dict:
                kernel_dict[name] = []

            if api not in kernel_dict[name]:
                kernel_dict[name].append(api)

        return kernel_dict

    def _create_combinations(self):
        combinations = []
        processed_pairs = set()  # Track processed API pairs to avoid duplicates

        for name, apis in self.kernel_dict.items():
            if len(apis) > 1:
                for api1, api2 in itertools.permutations(apis, 2):
                    # Skip if this kernel is in the ignore list
                    if f'{name}_{api1}_{api2}' in self.ignore_kernels:
                        continue

                    # Skip if we've already processed this API pair for this kernel
                    pair_key = f'{name}_{api1}_{api2}'
                    if pair_key in processed_pairs:
                        continue

                    # Add to combinations and mark as processed
                    combinations.append((name, api1, api2))
                    processed_pairs.add(pair_key)
            else:
                api1 = apis[0]
                api2 = apis[0]
                if f'{name}_{api1}_{api2}' in self.ignore_kernels:
                    continue
                combinations.append((name, api1, api2))

        return combinations

    def filter_kernels_with_mismatched_files(self):
        """
        Keep only combinations where:
        - api2 is 'cuda' or 'omp' (unless self.to_api is specified)
        - api1 matches self.from_api if specified
        - api2 matches self.to_api if specified
        - Both code1 and code2 exist and are dicts of length 1
        """
        filtered_combinations = []
        for name, api1, api2 in self.combinations:
            # Filter by from_api if specified
            if self.from_api and "all" not in self.from_api and api1 not in self.from_api:
                continue

            # Filter by to_api if specified
            if self.to_api and "all" not in self.to_api and api2 not in self.to_api:
                continue

            # Default filter for api2 if to_api is not specified
            if not self.to_api and api2 not in ('cuda', 'omp'):
                continue

            code1 = self.dataset.get(f"{name}-{api1}")
            code2 = self.dataset.get(f"{name}-{api2}")
            if (
                isinstance(code1, dict) and len(code1) == 1 and
                isinstance(code2, dict) and len(code2) == 1
            ):
                filtered_combinations.append((name, api1, api2))
                if len(filtered_combinations) == 600:
                    break

                # if (
                #     self.count_tokens(code1) <= 5000 and
                #     self.count_tokens(code2) <= 5000
                # ):
                #     filtered_combinations.append((name, api1, api2))

        print(f"Filtered combinations: {len(filtered_combinations)}")
        self.combinations = filtered_combinations

    def count_tokens(self, code):
        if isinstance(code, dict):
            # Get the first (and only) value from the dict
            code = next(iter(code.values()))
        return len(code.split())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                (
                    name,
                    api1,
                    self.dataset.get(f'{name}-{api1}'),
                    api2,
                    self.dataset.get(f'{name}-{api2}')
                )
                for name, api1, api2 in self.combinations[index]
            ]
        else:
            name, api1, api2 = self.combinations[index]
            code1 = self.dataset.get(f'{name}-{api1}')
            code2 = self.dataset.get(f'{name}-{api2}')
            return name, api1, code1, api2, code2

    def __len__(self):
        return len(self.combinations)
