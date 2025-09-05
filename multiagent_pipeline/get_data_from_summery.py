import pandas as pd
import sys
import argparse
# from collections import defaultdict

def analyze_compilation_data(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Group by source and target API
        api_pairs = df.groupby(['Source API', 'Target API'])
        
        # Store results
        results = []
        
        # Process each API pair
        for (source_api, target_api), group in api_pairs:
            total_count = len(group)
            
            # Skip if "Not Processed"
            processed_group = group[group['Compilation Status'] != 'Not Processed']
            if len(processed_group) == 0:
                continue
                
            # Calculate compilation success rate
            success_count = len(processed_group[processed_group['Compilation Status'] == 'Success'])
            if len(processed_group) > 0:
                success_rate = (success_count / len(processed_group)) * 100
            else:
                success_rate = 0
            
            # For successful compilations, calculate initial code vs subsequent attempts
            if success_count > 0:
                successful_compilations = processed_group[processed_group['Compilation Status'] == 'Success']
                initial_code_count = len(successful_compilations[successful_compilations['Compilation Attempt'] == 'Initial code (0)'])
                attempt_2_count = len(successful_compilations[successful_compilations['Compilation Attempt'] == 'Attempt 2'])
                attempt_3_count = len(successful_compilations[successful_compilations['Compilation Attempt'] == 'Attempt 3'])
                
                initial_code_rate = (initial_code_count / len(processed_group))
                attempt_2_rate = (attempt_2_count / len(processed_group))
                attempt_3_rate = (attempt_3_count / len(processed_group))
                subsequent_attempts_rate = attempt_2_rate + attempt_3_rate
            else:
                initial_code_rate = 0
                attempt_2_rate = 0
                attempt_3_rate = 0
                subsequent_attempts_rate = 0
            
            results.append({
                'Source API': source_api,
                'Target API': target_api,
                'Total Processed': len(processed_group),
                'Success Count': success_count,
                'Success Rate (%)': round(success_rate, 3),
                'Initial Code ': round(initial_code_rate, 3),
                'Attempt 2': round(attempt_2_rate, 3),
                'Attempt 3': round(attempt_3_rate, 3),
                'Subsequent Attempts (%)': round(subsequent_attempts_rate, 2)
            })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def main(file_path=None):
    if file_path is None:
        print("Usage: python get_data_from_summery.py <csv_file_path>")
        return
        
    results = analyze_compilation_data(file_path)
    
    if results is not None:
        # Display results
        print("\nCompilation Success Rates by API Pair:")
        print("=====================================")
        for _, row in results.iterrows():
            print(f"{row['Source API']} → {row['Target API']}:")
            print(f"  Total Processed: {row['Total Processed']}")
            print(f"  Success Rate: {row['Success Rate (%)']}%")
            print(f"  Initial Code: {row['Initial Code ']}%")
            print(f"  Attempt 2: {row['Attempt 2']}%")
            print(f"  Attempt 3: {row['Attempt 3']}%")
            print("-------------------------------------")
        
        # Save results to CSV
        output_file = file_path.replace('.csv', '_analysis.csv')
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Create the summary of compilation rates in Excel formula format
        # Order: ser-omp, ser-cuda, omp-cuda, cuda-omp, hip-omp, hip-cuda, sycl-omp, sycl-cuda
        api_pairs_order = [
            ('serial', 'omp'), ('serial', 'cuda'), 
            ('omp', 'cuda'), ('cuda', 'omp'),
            ('hip', 'omp'), ('hip', 'cuda'),
            ('sycl', 'omp'), ('sycl', 'cuda')
        ]
        
        excel_formulas = []
        excel_formulas_initial_code = []
        for source, target in api_pairs_order:
            pair_data = results[(results['Source API'] == source) & (results['Target API'] == target)]
            if not pair_data.empty:
                success_count = pair_data['Success Count'].values[0]
                total_processed = pair_data['Total Processed'].values[0]
                initial_code_count = int(round(pair_data['Initial Code '].values[0] * total_processed))
                excel_formulas.append(f"=ROUND({success_count}/{total_processed},3)")
                excel_formulas_initial_code.append(f"=ROUND({initial_code_count}/{total_processed},3)")
            else:
                excel_formulas.append("=ROUND(0/0,3)")
                excel_formulas_initial_code.append("=ROUND(0/0,3)")
        
        print("\nExcel Formulas for Compilation Rates:")
        print("\t".join(excel_formulas))
        print("\nExcel Formulas for Initial Code Rates:")
        print("\t".join(excel_formulas_initial_code))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze compilation data from a summary CSV file.")
    parser.add_argument("csv_file_path", help="Path to the summary CSV file")
    args = parser.parse_args()
    
    main(args.csv_file_path)
