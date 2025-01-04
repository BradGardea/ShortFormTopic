import pandas as pd

def update_post_type_based_on_empty_column(csv_file, target_column):
    """
    Update 'post_type' in the CSV based on whether a specific column is empty.
    
    Parameters:
    - csv_file (str): Path to the CSV file.
    - target_column (str): The column to check for empty values.
    - new_post_type (str): The value to set in the 'post_type' column if the target column is empty.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Check if the target column exists in the DataFrame
        if target_column not in df.columns:
            print(f"Column '{target_column}' not found in the CSV file.")
            return
        
        # Check if 'post_type' column exists; if not, create it
        if 'post_type' not in df.columns:
            df['post_type'] = None  # Create a 'post_type' column with default None values
        
        # Update 'post_type' based on the condition where the target column is empty
        df.loc[df[target_column].isna() | (df[target_column] == ''), 'post_type'] = "video"

        # Update 'post_type' to 'text' where the target column is not NaN and not an empty string
        df.loc[
            ~df[target_column].isna() & (df[target_column] != '') & 
            (df["title"].str.contains("AITA") | df["title"].str.contains("WIBTA")), 
            'post_type'
        ] = "AITA"

        df.loc[~df[target_column].isna() & (df[target_column] != '') & ("AITA" not in df["title"]) & ("WIBTA" in df["title"]), 'post_type'] = "text"



        
        # Save the updated DataFrame back to the CSV
        df.to_csv(csv_file, index=False)
        print(f"Updated 'post_type' in the CSV file based on empty '{target_column}' column.")

    except FileNotFoundError:
        print(f"File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Example usage:
    csv_file_path = 'data/reddit_posts.csv'  # Path to your CSV file
    target_column_name = 'content'           # Column to check for emptiness

    update_post_type_based_on_empty_column(csv_file_path, target_column_name)
