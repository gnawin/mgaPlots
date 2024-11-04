import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

def filter_technologies(file_path: str, indicator: str, techs_to_plot: Optional[List[str]] = None, all_techs: bool = True) -> pd.DataFrame:
    """
    Filters the dataset based on the selected technologies.

    Parameters:
    file_path (str): Path to the CSV file.
    techs_to_plot (List[str]): List of technologies to include in the filtered dataset.

    Returns:
    pd.DataFrame: Filtered dataset based on the selected technologies.
    """
    df = pd.read_csv(file_path)
    if all_techs:
        techs = df['techs'].unique().tolist()
    else:
        techs = techs_to_plot    

    # Strip any leading or trailing whitespace from the color values
    df['colors'] = df['colors'].str.strip()

    # Filter technologies based on the specified prefixes
    prefixes = ('pp_nuclear', 'wind', 'pv')
    df_filtered = df[df['techs'].str.startswith(prefixes)]

    df_filtered[df_filtered['techs'].isin(techs)][['techs', 'locs', 'spores', f'{indicator}', 'colors']]

    # Assuming df_filtered is already defined
    tech_counts = df_filtered[df_filtered['techs'].isin(techs)].groupby('techs').size()

    # Print the number of rows for each technology
    print("Number of rows for each technology:")
    print(tech_counts)

     # Identify repeated spores and rename them starting from 51
    def rename_spores(group):
        spore_counts = group['spores'].value_counts()
        repeated_spores = spore_counts[spore_counts > 1].index
        next_spore = 51
        for spore in repeated_spores:
            indices = group[group['spores'] == spore].index
            for i in range(1, len(indices)):
                group.at[indices[i], 'spores'] = next_spore
                next_spore += 1
        return group

    df_filtered = df_filtered.groupby('techs').apply(rename_spores)


    # Debug: Print the DataFrame after renaming spores
    print("DataFrame after renaming spores:")
    print(df_filtered.head())

    # Print the unique number of spores
    print(f"Unique number of spores in df_filtered: {df_filtered['spores'].nunique()}")
    
    
    
    
    return df_filtered

def create_strip_plot(df_filtered_techs: pd.DataFrame, indicator: str, save_path: str) -> None:
    """
    Creates a swarm plot using seaborn for the filtered dataset.

    Parameters:
    df_filtered_techs (pd.DataFrame): The filtered dataset with the selected technologies.
    """
    
    plt.figure(figsize=(12, 6))
    sns.stripplot(x='techs', y=f'{indicator}', data=df_filtered_techs, alpha=0.5, jitter=True, palette=df_filtered_techs['colors'].unique())
    plt.xlabel('Technologies')
    plt.xticks(rotation=45) 
    # plt.ylabel('Energy Capacity')
    # plt.title('Energy Capacity of Selected Technologies')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def create_box_plot(df_filtered_techs: pd.DataFrame, indicator: str, save_path: str) -> None:
    """
    Creates a swarm plot using seaborn for the filtered dataset.

    Parameters:
    df_filtered_techs (pd.DataFrame): The filtered dataset with the selected technologies.
    """
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='techs', y=f'{indicator}', data=df_filtered_techs, palette=df_filtered_techs['colors'].unique())
    plt.xlabel('Technologies')
    plt.xticks(rotation=45) 
    # plt.ylabel('Energy Capacity')
    # plt.title('Energy Capacity of Selected Technologies')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def process_and_plot(input_csv: str, output_dir: str, indicator: str) -> None:
    """
    Processes the input CSV file, filters the data, and generates strip and box plots.

    Parameters:
    input_csv (str): Path to the input CSV file.
    output_dir (str): Directory to save the output plots.
    indicator (str): The name of the indicator to plot.
    """
    # Get the absolute path to the CSV file
    absolute_path = os.path.abspath(input_csv)

    # Process the CSV file to filter the data
    df_filtered_techs = filter_technologies(absolute_path, indicator, all_techs=True)

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the paths to save the plots
    save_path_strip = os.path.join(output_dir, f'{indicator}_strip_plot.png')
    save_path_box = os.path.join(output_dir, f'{indicator}_box_plot.png')

    # Generate and save the strip plot
    create_strip_plot(df_filtered_techs, indicator, save_path_strip)

    # Generate and save the box plot
    create_box_plot(df_filtered_techs, indicator, save_path_box)

if __name__ == "__main__":
    process_and_plot('inputs/50_spores_PC_floris/energy_cap.csv', 'outputs/50_spores_PC_floris', 'energy_cap')
