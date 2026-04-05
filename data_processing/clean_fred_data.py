def clean_macro_data(macro_df):
    print("Cleaning macroeconomic data...")
    # Drop rows with missing values
    macro_df = macro_df.dropna()
    # Flatten columns and convert to lowercase
    macro_df.columns = [col.lower() for col in macro_df.columns]
    
    macro_df['date'] = macro_df['date'].dt.date  # Convert datetime to date for easier handling
    
    return macro_df