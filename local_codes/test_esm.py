import torch
import time
import psutil
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, EsmModel
import gc
import pandas as pd
from torch.cuda import memory_stats, memory_summary
import matplotlib.pyplot as plt


def generate_dummy_data(seq_length, batch_size):
    """Generate dummy protein sequences for testing."""
    vocab = list('ACDEFGHIKLMNPQRSTVWY')
    sequences = [''.join(np.random.choice(vocab, seq_length)) for _ in range(batch_size)]
    labels = torch.randint(0, 2, (batch_size,))
    return sequences, labels


def measure_gpu_memory():
    """Measure current GPU memory usage."""
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
        'cached': torch.cuda.memory_reserved() / 1024**2      # MB
    }


class ESMBenchmark:
    def __init__(self):
        self.models = {
            'esm2_t6_8M': 'facebook/esm2_t6_8M_UR50D',
            'esm2_t12_35M': 'facebook/esm2_t12_35M_UR50D',
            'esm2_t30_150M': 'facebook/esm2_t30_150M_UR50D',
            'esm2_t33_650M': 'facebook/esm2_t33_650M_UR50D',
            'esm2_t36_3B': 'facebook/esm2_t36_3B_UR50D',
            'esm1b_t33_650M': "facebook/esm1b_t33_650M_UR50S"
        }
        self.sequence_lengths = [512, 1024]
        self.batch_sizes = [16, 32, 64]
        self.results = []
    
    def benchmark_inference(self, model, tokenizer, sequences, batch_size):
        model.eval()
        """Measure inference speed and memory."""
        inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)
        
        # Measure memory before
        mem_before = measure_gpu_memory()
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):  # Run multiple times for more stable measurements
                _ = model(**inputs)
        end_time = time.time()

        torch.cuda.synchronize()
        
        # Measure memory after
        mem_after = measure_gpu_memory()
        
        return {
            'avg_time': (end_time - start_time) / 100,
            'memory_increase': mem_after['allocated'] - mem_before['allocated']
        }
    
    def benchmark_training(self, model, tokenizer, sequences, labels, batch_size):
        model.train()
        """Measure training speed and memory."""
        inputs = tokenizer(sequences, return_tensors="pt", padding=True).to(device)
        labels = labels.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Measure memory before
        mem_before = measure_gpu_memory()
        
        # Measure training time
        start_time = time.time()
        for _ in range(100):  # Fewer iterations for training due to higher memory usage
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.pooler_output, labels)
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
    
        end_time = time.time()
        
        # Measure memory after
        mem_after = measure_gpu_memory()
        
        return {
            'avg_time': (end_time - start_time) / 100,
            'memory_increase': mem_after['allocated'] - mem_before['allocated']
        }
    
    def run_benchmarks(self):
        """Run comprehensive benchmarks across models and configurations."""
        for model_name, model_path in self.models.items():
            print(f"\nBenchmarking {model_name}...")
            
            try:
                # Load model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = EsmModel.from_pretrained(model_path).to(device)
                
                for seq_length in self.sequence_lengths:
                    for batch_size in self.batch_sizes:
                        try:
                            # Generate test data
                            sequences, labels = generate_dummy_data(seq_length, batch_size)
                            
                            torch.cuda.empty_cache()
                            gc.collect()
                            
                            # Run benchmarks
                            inf_metrics = self.benchmark_inference(model, tokenizer, sequences, batch_size)
                            train_metrics = self.benchmark_training(model, tokenizer, sequences, labels, batch_size)
                            
                            self.results.append({
                                'model': model_name,
                                'sequence_length': seq_length,
                                'batch_size': batch_size,
                                'inference_time': round(inf_metrics['avg_time'], 4),
                                'inference_memory': round(inf_metrics['memory_increase'], 4),
                                'training_time': round(train_metrics['avg_time'], 4),
                                'training_memory': round(train_metrics['memory_increase'], 4)
                            })
                            
                        except RuntimeError as e:
                            print(f"Error with {model_name}, seq_length={seq_length}, batch_size={batch_size}: {e}")
                            continue
                
                # Clear model from GPU
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                continue
    
    def plot_results(self):
        """Generate visualization of benchmark results."""
        df = pd.DataFrame(self.results)
        
        return df

device = 'cuda:0'
benchmark = ESMBenchmark()
benchmark.run_benchmarks()
results_df = benchmark.plot_results()

# Save results to CSV
results_df.to_csv('esm_benchmark_results.csv', index=False)
print("\nBenchmark complete! Results saved to CSV and plots generated.")