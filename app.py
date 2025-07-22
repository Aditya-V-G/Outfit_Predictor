print('DEBUG: Starting app.py')
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ml_recommender import OutfitRecommender
import os

print('DEBUG: Creating Flask app')
app = Flask(__name__)
CORS(app)

print('DEBUG: Initializing OutfitRecommender')
# Initialize ML recommender
recommender = OutfitRecommender()
print('DEBUG: OutfitRecommender initialized')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log incoming request
        print("\n=== Incoming Request ===")
        print(f"Headers: {request.headers}")
        print(f"Content-Type: {request.content_type}")
        
        # Get JSON data
        data = request.get_json()
        print(f"Request data: {data}")
        
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        gender = data.get('gender', 'male')
        print(f"Gender: {gender}")
        
        # Ensure recommender is initialized
        if not recommender.model:
            print("Initializing recommender model...")
            recommender.load_and_process_data()
            recommender.train_model()
        
        # Get traits based on personality type
        print("\n=== Processing Request ===")
        print(f"Request keys: {data.keys()}")
        
        traits = {}
        
        # Check for MBTI type first
        if 'mbti_type' in data:
            print("MBTI type detected in request")
            mbti_type = data.get('mbti_type', '').upper()
            print(f"MBTI type: {mbti_type}")
            
            if not mbti_type:
                return jsonify({'error': 'MBTI type is required'}), 400
            
            # Convert MBTI to traits
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
            
            # Extract personality traits from MBTI type
            for char in mbti_type:
                if char in mbti_traits:
                    traits.update(mbti_traits[char])
            
            print(f"Converted MBTI to traits: {traits}")
        
        else:
            # Check for direct Big Five traits in the root of the request
            big_five_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
            
            # Default values (middle of the slider range)
            default_value = 0.6  # 3/5 = 0.6
            traits = {}  # Initialize with empty traits
            
            # Process each trait, using default if not provided
            for trait in big_five_traits:
                if trait in data:
                    try:
                        value = data[trait]
                        if isinstance(value, str):
                            value = float(value)
                        traits[trait] = value
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid value for {trait}: {data.get(trait)}, using default. Error: {str(e)}")
                        traits[trait] = default_value
                else:
                    print(f"Warning: {trait} not provided, using default value: {default_value}")
                    traits[trait] = default_value
            
            print(f"Using Big Five traits (defaults for missing values): {traits}")
        
        # Get outfit recommendation based on traits
        try:
            print("\n=== Getting Outfit Recommendation ===")
            print(f"Using traits: {traits}")
            print(f"Gender: {gender}")
            print(f"Model loaded: {hasattr(recommender, 'model') and recommender.model is not None}")
            
            # Get outfit recommendation with error handling
            try:
                # Get style prediction
                predicted_style = recommender.predict_outfit(traits, gender=gender)
                print(f"Predicted style: {predicted_style}")
                # Get full outfit details from predicted style
                recommended_outfit = recommender.get_recommended_outfit(predicted_style, gender)
                print(f"Recommended outfit: {recommended_outfit}")
                if not recommended_outfit:
                    print("Warning: Empty outfit recommendation received")
                    recommended_outfit = {
                        'top': 'default top',
                        'bottom': 'default bottom',
                        'shoes': 'default shoes',
                        'accessories': 'default accessories'
                    }
                response_data = {
                    'status': 'success',
                    'outfit': recommended_outfit,
                    'style': predicted_style,
                    'gender': gender,
                    'traits': traits
                }
                print(f"Sending response: {response_data}")
                return jsonify(response_data)
                
            except Exception as e:
                print(f"Error in predict_outfit: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Return a default outfit if prediction fails
                default_outfit = {
                    'top': 'classic t-shirt',
                    'bottom': 'jeans',
                    'shoes': 'sneakers',
                    'accessories': 'watch',
                    'note': 'Default outfit - prediction failed'
                }
                return jsonify({
                    'status': 'success',
                    'outfit': default_outfit,
                    'gender': gender,
                    'traits': traits,
                    'warning': 'Using default outfit due to prediction error'
                })
        
        except Exception as e:
            return jsonify({
                'error': str(e)
            }), 500
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
