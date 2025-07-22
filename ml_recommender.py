import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import os

class OutfitRecommender:
    def __init__(self, data_path='data/processed_data.json'):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.load_and_process_data()

    def load_and_process_data(self):
        """Load and preprocess the data"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            
        # Create feature and target data
        features = []
        targets = []
        
        # Convert MBTI types to personality traits
        mbti_traits = {
            'I': {'Extraversion': 0},
            'E': {'Extraversion': 1},
            'N': {'Openness': 1},
            'S': {'Openness': 0},
            'T': {'Agreeableness': 0},
            'F': {'Agreeableness': 1},
            'J': {'Conscientiousness': 1},
            'P': {'Conscientiousness': 0}
        }
        
        # Create training data from MBTI styles
        for mbti_type, style in data['mbti_styles'].items():
            # Extract personality traits from MBTI type
            traits = {}
            for char in mbti_type:
                traits.update(mbti_traits[char])
            
            # Add gender variations
            for gender in ['male', 'female']:
                features.append({**traits, 'Gender': gender})
                targets.append(style)
        
        # Convert to DataFrame
        self.df = pd.DataFrame(features)
        self.df['Style'] = targets
        
        # Encode categorical variables
        self.label_encoders['Gender'] = LabelEncoder()
        self.df['Gender'] = self.label_encoders['Gender'].fit_transform(self.df['Gender'])
        
        # Split features and target
        self.X = self.df.drop('Style', axis=1)
        self.y = self.df['Style']
        
        # Split train-test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model(self):
        """Train a Random Forest classifier"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate model
        y_pred = self.model.predict(self.X_test)
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.2f}")
        print("Feature Importances:")
        for feature, importance in zip(self.X_train.columns, self.model.feature_importances_):
            print(f"{feature}: {importance:.3f}")
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

    def predict_outfit(self, personality_traits, gender):
        """Predict outfit style based on personality traits and gender"""
        # Prepare input data
        input_data = pd.DataFrame([{
            'Extraversion': personality_traits.get('Extraversion', 0),
            'Openness': personality_traits.get('Openness', 0),
            'Agreeableness': personality_traits.get('Agreeableness', 0),
            'Conscientiousness': personality_traits.get('Conscientiousness', 0),
            'Gender': gender
        }])
        
        # Transform gender
        input_data['Gender'] = self.label_encoders['Gender'].transform(input_data['Gender'])
        
        # Predict style
        predicted_style = self.model.predict(input_data)[0]
        
        return predicted_style

    def get_recommended_outfit(self, style, gender):
        """Get outfit recommendations based on style and gender"""
        # Load outfit database
        with open(self.data_path, 'r') as f:
            data = json.load(f)
            outfit_database = data['outfit_database']
            
        # Get items from database
        items = outfit_database[style]
        
        # Initialize outfit with default values
        outfit = {
            'top': 'Basic T-Shirt',
            'bottom': 'Black Jeans',
            'shoes': 'Classic Sneakers',
            'accessories': ['Simple Watch']
        }
        
        # Get random items based on gender
        if gender == 'male':
            outfit['top'] = np.random.choice(items.get('tops_male', ['Basic T-Shirt']))
            outfit['bottom'] = np.random.choice(items.get('bottoms_male', ['Black Jeans']))
            outfit['shoes'] = np.random.choice(items.get('footwear_male', ['Classic Sneakers']))
            # Filter out necklaces and earrings for male outfits
            accessories = [acc for acc in items.get('accessories', ['Simple Watch']) 
                          if 'necklace' not in acc.lower() and 'earring' not in acc.lower()]
            outfit['accessories'] = [np.random.choice(accessories or ['Simple Watch'])]
        else:
            outfit['top'] = np.random.choice(items.get('tops_female', ['Basic T-Shirt']))
            outfit['bottom'] = np.random.choice(items.get('bottoms_female', ['Black Jeans']))
            outfit['shoes'] = np.random.choice(items.get('footwear_female', ['Classic Sneakers']))
            outfit['accessories'] = [np.random.choice(items.get('accessories', ['Simple Watch']))]
        
        # Remove any 'None' values
        for key in outfit:
            if outfit[key] == 'None':
                if key == 'top':
                    outfit[key] = 'Basic T-Shirt'
                elif key == 'bottom':
                    outfit[key] = 'Black Jeans'
                elif key == 'shoes':
                    outfit[key] = 'Classic Sneakers'
                elif key == 'accessories':
                    outfit[key] = ['Simple Watch']
        
        return outfit

    def get_recommended_outfit_from_traits(self, traits, gender):
        """Get outfit recommendation based on personality traits"""
        # Convert traits to style using ML model
        predicted_style = self.predict_outfit(traits, gender)
        
        # Get outfit based on predicted style
        return self.get_recommended_outfit(predicted_style, gender)
