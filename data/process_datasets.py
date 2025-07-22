import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import zipfile
import os

def load_mbti_dataset():
    """Load and process the MBTI dataset"""
    # Extract the zip file
    with zipfile.ZipFile(r'd:\outfitgen\(MBTI) Myers-Briggs Personality Type Dataset.zip', 'r') as zip_ref:
        zip_ref.extractall(r'd:\outfitgen\data\mbti')

    # Load the dataset
    mbti_df = pd.read_csv(r'd:\outfitgen\data\mbti\mbti_1.csv')
    
    # Process the data
    mbti_df['type'] = mbti_df['type'].str.upper()
    
    # Create personality type mapping
    mbti_styles = {
        'INTJ': 'Minimalist',
        'INTP': 'Casual',
        'ENTJ': 'Formal',
        'ENTP': 'Bohemian',
        'INFJ': 'Vintage',
        'INFP': 'Bohemian',
        'ENFJ': 'Classic',
        'ENFP': 'Bohemian',
        'ISTJ': 'Classic',
        'ISFJ': 'Vintage',
        'ESTJ': 'Formal',
        'ESFJ': 'Classic',
        'ISTP': 'Casual',
        'ISFP': 'Bohemian',
        'ESTP': 'Casual',
        'ESFP': 'Bohemian'
    }
    
    return mbti_df, mbti_styles

# Style rules are now based solely on MBTI types
style_rules = {
    'Minimalist': {},
    'Bohemian': {},
    'Classic': {},
    'Formal': {},
    'Vintage': {},
    'Casual': {}
}

def create_outfit_database():
    """Create a database of fashion items with gender-specific options"""
    outfit_database = {
        'Minimalist': {
            'tops_male': ['White Shirt', 'Black T-Shirt', 'Gray Sweater'],
            'tops_female': ['White Shirt', 'Black T-Shirt', 'Gray Sweater', 'Sweater Dress'],
            'bottoms_male': ['Black Jeans', 'Khaki Trousers', 'Black Trousers'],
            'bottoms_female': ['Black Jeans', 'Khaki Trousers', 'Black Trousers', 'Black Skirt'],
            'footwear_male': ['Black Oxfords', 'Black Sneakers', 'Black Loafers', 'Brown Brogues', 'White Sneakers', 'Tan Desert Boots',
                              'Black Chelsea Boots', 'Brown Monk Straps', 'Tan Chukka Boots', 'White Leather Sneakers',
                              'Black Combat Boots', 'Tan Hiking Boots'],
            'footwear_female': ['Black Oxfords', 'Black Sneakers', 'Black Loafers', 'Black Heels', 'Brown Heels', 'Tan Wedges', 
                               'White Sneakers', 'Red Heels', 'Black Pumps', 'Nude Heels', 'Tan Ankle Boots',
                               'White Leather Sneakers', 'Black Combat Boots', 'Tan Hiking Boots', 'Black Sandals',
                               'Tan Espadrilles', 'Black Mules', 'Tan Platform Sandals'],
            'footwear': ['Black Oxfords', 'Black Sneakers', 'Black Loafers', 'White Sneakers', 'Tan Desert Boots',
                         'Tan Chukka Boots', 'White Leather Sneakers', 'Tan Hiking Boots'],
            'footwear_casual': ['Black Sneakers', 'White Sneakers', 'Tan Desert Boots', 'Tan Chukka Boots',
                               'White Leather Sneakers', 'Tan Hiking Boots', 'Black Combat Boots'],
            'footwear_formal': ['Black Oxfords', 'Black Loafers', 'Brown Brogues', 'Black Chelsea Boots',
                               'Brown Monk Straps', 'Black Pumps', 'Nude Heels'],
            'accessories': ['Classic Watch', 'Leather Belt', 'Simple Sunglasses', 'Silver Bracelet', 'Gold Chain', 'Leather Wallet', 'Diamond Studs']
        },
        'Casual': {
            'tops_male': ['T-Shirt', 'Hoodie', 'Polo Shirt'],
            'tops_female': ['T-Shirt', 'Hoodie', 'Polo Shirt', 'Crop Top'],
            'bottoms_male': ['Jeans', 'Cargo Pants', 'Shorts'],
            'bottoms_female': ['Jeans', 'Cargo Pants', 'Shorts', 'Skirt'],
            'footwear_male': ['Sneakers', 'Flip Flops', 'Boots'],
            'footwear_female': ['Sneakers', 'Flip Flops', 'Boots', 'Sneaker Boots'],
            'accessories': ['Cap', 'Sunglasses', 'Casual Watch', 'Sport Earphones']
        },
        'Formal': {
            'tops_male': ['White Shirt', 'Blue Blazer', 'Black Suit Jacket'],
            'tops_female': ['White Shirt', 'Blue Blazer', 'Black Suit Jacket', 'Formal Dress'],
            'bottoms_male': ['Black Trousers', 'Navy Suit Pants', 'Gray Trousers'],
            'bottoms_female': ['Black Trousers', 'Navy Suit Pants', 'Gray Trousers', 'Formal Skirt'],
            'footwear_male': ['Black Oxfords', 'Black Sneakers', 'Black Loafers', 'Brown Brogues', 'White Sneakers', 'Tan Desert Boots', 'Black Chelsea Boots', 'Brown Monk Straps'],
            'footwear_female': ['Black Oxfords', 'Black Sneakers', 'Black Loafers', 'Black Heels', 'Brown Heels', 'Tan Wedges', 'White Sneakers', 'Red Heels', 'Black Pumps', 'Nude Heels'],
            'accessories': ['Classic Watch', 'Leather Belt', 'Simple Sunglasses', 'Silver Bracelet', 'Gold Chain', 'Leather Wallet', 'Pocket Square', 'Cufflinks']
        },
        'Bohemian': {
            'tops_male': ['Floral Shirt', 'Silk Top', 'Crochet Top'],
            'tops_female': ['Floral Blouse', 'Silk Top', 'Crochet Top', 'Boho Dress'],
            'bottoms_male': ['Cargo Pants', 'Jeans', 'Shorts'],
            'bottoms_female': ['Maxi Skirt', 'Boho Pants', 'Jeans', 'Boho Skirt'],
            'footwear_male': ['Sandals', 'Boho Boots', 'Espadrilles'],
            'footwear_female': ['Sandals', 'Boho Boots', 'Espadrilles', 'Boho Heels'],
            'accessories': ['Statement Necklace', 'Feather Earrings', 'Floppy Hat']
        },
        'Vintage': {
            'tops_male': ['Polo Shirt', 'Button-Up Shirt', 'Cropped Sweater'],
            'tops_female': ['Polo Shirt', 'Button-Up Shirt', 'Cropped Sweater', 'Vintage Dress'],
            'bottoms_male': ['Slim Jeans', 'Chino Pants', 'Cargo Shorts'],
            'bottoms_female': ['Slim Jeans', 'Chino Pants', 'Cargo Shorts', 'Vintage Skirt'],
            'footwear_male': ['Loafers', 'Sneakers', 'Boots'],
            'footwear_female': ['Loafers', 'Sneakers', 'Boots', 'Vintage Heels'],
            'accessories': ['Classic Watch', 'Leather Belt', 'Vintage Sunglasses']
        },
        'Classic': {
            'tops_male': ['White Shirt', 'Blue Blazer', 'Navy Sweater'],
            'tops_female': ['White Shirt', 'Blue Blazer', 'Navy Sweater', 'Classic Dress'],
            'bottoms_male': ['Black Trousers', 'Khaki Pants', 'Chino Pants'],
            'bottoms_female': ['Black Trousers', 'Khaki Pants', 'Chino Pants', 'Classic Skirt'],
            'footwear_male': ['Black Oxfords', 'Brown Loafers', 'Black Brogues'],
            'footwear_female': ['Black Oxfords', 'Brown Loafers', 'Black Brogues', 'Classic Heels'],
            'accessories': ['Classic Watch', 'Leather Belt', 'Silk Tie', 'Pearl Necklace']
        }
    }

    return outfit_database

def main():
    # Load and process datasets
    mbti_df, mbti_styles = load_mbti_dataset()
    
    # Create outfit database
    outfit_database = create_outfit_database()
    
    # Save processed data
    processed_data = {
        'mbti_styles': mbti_styles,
        'style_rules': style_rules,
        'outfit_database': outfit_database
    }
    
    with open('processed_data.json', 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()
