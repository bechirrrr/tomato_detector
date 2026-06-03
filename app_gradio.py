"""
🌿 Interface Gradio — Détection de Maladies des Plantes (Tomate)
Lancer avec : python app_gradio.py
"""

import os
import pickle
import numpy as np
import cv2
# ── Images prétraitement ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import gradio as gr
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops

# ─── Configuration ────────────────────────────────────────────────
IMG_SIZE    = (128, 128)
CLASS_NAMES = ['Tomato Healthy', 'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Bacterial Spot']
CLASS_INFO  = {
    'Tomato Healthy': {
        'emoji': '✅',
        'color': '#2ecc71',
        'description': 'La feuille est saine. Aucune maladie détectée.',
        'conseil': 'Continuez l\'entretien régulier : arrosage modéré, bonne exposition au soleil.',
        'gravite': 'Aucune'
    },
    'Tomato Early Blight': {
        'emoji': '⚠️',
        'color': '#e67e22',
        'description': 'Brûlure précoce (Alternaria solani). Taches brunes concentriques sur les feuilles.',
        'conseil': 'Appliquer un fongicide à base de cuivre. Retirer les feuilles infectées.',
        'gravite': 'Modérée'
    },
    'Tomato Late Blight': {
        'emoji': '🔴',
        'color': '#e74c3c',
        'description': 'Brûlure tardive (Phytophthora infestans). Maladie très agressive et contagieuse.',
        'conseil': 'Traitement fongicide urgent. Isoler les plants touchés immédiatement.',
        'gravite': 'Élevée'
    },
    'Tomato Bacterial Spot': {
        'emoji': '🟡',
        'color': '#f39c12',
        'description': 'Tache bactérienne (Xanthomonas). Petites taches noires entourées de jaune.',
        'conseil': 'Pulvériser de la bouillie bordelaise. Éviter l\'arrosage par aspersion.',
        'gravite': 'Modérée'
    }
}

# ─── Chargement des modèles ───────────────────────────────────────
MODELS_DIR = Path('saved_models')

def load_models():
    models = {}
    try:
        with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
        with open(MODELS_DIR / 'random_forest.pkl', 'rb') as f:
            models['Random Forest'] = pickle.load(f)
        with open(MODELS_DIR / 'svm.pkl', 'rb') as f:
            models['SVM'] = pickle.load(f)
        with open(MODELS_DIR / 'knn.pkl', 'rb') as f:
            models['KNN'] = pickle.load(f)
        print("✅ Modèles ML chargés")
    except Exception as e:
        print(f"⚠️  Erreur chargement modèles ML : {e}")

    # Charger CNN (avec contournement de l'erreur TrueDivide)
    try:
        import tensorflow as tf

        @tf.keras.utils.register_keras_serializable()
        class TrueDivide(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            def call(self, inputs):
                # Divise simplement par 255.0 (cas classique de normalisation)
                return tf.math.truediv(inputs, 255.0)

        models['CNN'] = tf.keras.models.load_model(
            MODELS_DIR / 'cnn_model.h5',
            custom_objects={'TrueDivide': TrueDivide},
            compile=False
        )
        print("✅ CNN chargé")
    except Exception as e:
        print(f"⚠️  CNN non disponible : {e}")

    return models

MODELS = load_models()

# ─── Pipeline de features (identique au notebook) ────────────────
def extract_color_features(img_rgb, n_bins=32):
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img_rgb], [ch], None, [n_bins], [0, 256]).flatten()
        feats.extend(hist / (hist.sum() + 1e-6))
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    for ch in range(3):
        maxval = 180 if ch == 0 else 256
        hist   = cv2.calcHist([hsv], [ch], None, [n_bins], [0, maxval]).flatten()
        feats.extend(hist / (hist.sum() + 1e-6))
    return np.array(feats)

def extract_texture_features(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if gray.max() > 0:
        gray = (gray / gray.max() * 255).astype(np.uint8)
    glcm  = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                         levels=256, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        val = graycoprops(glcm, prop)
        feats.extend([val.mean(), val.std()])
    return np.array(feats)

def extract_shape_features(img_rgb):
    hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([90, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0.0, 0.0, 0.0, 0.0])
    cnt   = max(contours, key=cv2.contourArea)
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    circ  = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0
    coverage = area / (img_rgb.shape[0] * img_rgb.shape[1])
    return np.array([area, perim, circ, coverage])

def extract_all_features(img_rgb):
    return np.concatenate([
        extract_color_features(img_rgb),
        extract_texture_features(img_rgb),
        extract_shape_features(img_rgb)
    ])

# ─── Prétraitement visuel (pour affichage) ───────────────────────
def preprocess_for_display(img_rgb):
    """Retourne les étapes de prétraitement pour affichage."""
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    blurred = cv2.GaussianBlur(img_rgb, (5, 5), 0)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq      = clahe.apply(gray)

    # Segmentation Canny
    canny = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)

    # Masque vert HSV
    mask    = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255]))
    seg_hsv = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    return {
        'Grayscale':     cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        'HSV':           cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB),
        'Gaussian Blur': blurred,
        'CLAHE':         cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB),
        'Canny Edges':   cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB),
        'Segm. HSV':     seg_hsv,
    }

# ─── Fonction principale de prédiction ───────────────────────────
def predict(image, selected_model):
    if image is None:
        return "❌ Veuillez uploader une image.", None, None

    # Redimensionner
    img = cv2.resize(np.array(image), IMG_SIZE)
    img_rgb = img.copy()

    # ── Prédiction selon modèle choisi ──
    predicted_class = None
    confidence_dict = {}

    if selected_model == 'CNN' and 'CNN' in MODELS:
        img_norm = img_rgb.astype('float32') / 255.0
        probs = MODELS['CNN'].predict(img_norm[np.newaxis, ...], verbose=0)[0]
        predicted_class = CLASS_NAMES[np.argmax(probs)]
        confidence_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}

    elif selected_model in MODELS and 'scaler' in MODELS:
        feats = extract_all_features(img_rgb).reshape(1, -1)
        scaled = MODELS['scaler'].transform(feats)
        model = MODELS[selected_model]
        pred_idx = model.predict(scaled)[0]
        predicted_class = CLASS_NAMES[pred_idx]
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(scaled)[0]
            confidence_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(4)}
        else:
            confidence_dict = {CLASS_NAMES[i]: 1.0 if i == pred_idx else 0.0 for i in range(4)}
    else:
        return "⚠️ Modèle non disponible.", None, None

    # ── Résultat HTML ──
    info = CLASS_INFO[predicted_class]
    confidence = confidence_dict.get(predicted_class, 0) * 100

    result_html = f"""
    <div style="font-family:'Segoe UI',sans-serif; padding:20px; border-radius:12px;
                border-left:6px solid {info['color']}; background:#1a1a2e; color:#eee;">
        <h2 style="margin:0; color:{info['color']}; font-size:1.5em;">
            {info['emoji']} {predicted_class}
        </h2>
        <p style="margin:8px 0; color:#ccc;">{info['description']}</p>
        <hr style="border-color:#333; margin:12px 0;">
        <div style="display:flex; gap:30px; flex-wrap:wrap;">
            <div>
                <span style="color:#aaa; font-size:0.85em;">CONFIANCE</span><br>
                <span style="color:{info['color']}; font-size:1.8em; font-weight:bold;">{confidence:.1f}%</span>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.85em;">GRAVITÉ</span><br>
                <span style="font-size:1.4em; font-weight:bold;">{info['gravite']}</span>
            </div>
            <div>
                <span style="color:#aaa; font-size:0.85em;">MODÈLE</span><br>
                <span style="font-size:1.2em; font-weight:bold;">{selected_model}</span>
            </div>
        </div>
        <hr style="border-color:#333; margin:12px 0;">
        <p><b style="color:{info['color']};">💡 Conseil :</b> {info['conseil']}</p>
    </div>
    """

    # ── Probabilités HTML ──
    proba_html = "<div style='font-family:monospace; padding:10px; background:#111; border-radius:8px;'>"
    proba_html += "<b style='color:#aaa;'>Probabilités par classe :</b><br><br>"
    for cls, prob in sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True):
        bar_w = int(prob * 200)
        clr = CLASS_INFO[cls]['color']
        em = CLASS_INFO[cls]['emoji']
        proba_html += f"""
        <div style='margin:6px 0;'>
            <span style='color:#ccc; font-size:0.85em;'>{em} {cls}</span><br>
            <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:{bar_w}px; height:12px; background:{clr};
                            border-radius:6px; min-width:2px;'></div>
                <span style='color:{clr}; font-weight:bold;'>{prob*100:.1f}%</span>
            </div>
        </div>"""
    proba_html += "</div>"

    

    pp_steps = preprocess_for_display(img_rgb)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('Étapes de Prétraitement', color='white', fontsize=13, fontweight='bold')

    for ax, (title, step_img) in zip(axes.flatten(), pp_steps.items()):
        ax.imshow(step_img)
        ax.set_title(title, color='#aaa', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    tmp_path = os.path.join(tempfile.gettempdir(), 'preprocess_steps.png')
    fig.savefig(tmp_path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)

    from PIL import Image as PILImage
    pp_img = PILImage.open(tmp_path)

    return result_html, proba_html, pp_img

# ─── Interface Gradio ─────────────────────────────────────────────
def build_interface():
    available_models = [m for m in ['Random Forest', 'SVM', 'KNN', 'CNN'] if m in MODELS]
    if not available_models:
        available_models = ['Random Forest', 'SVM', 'KNN']

    css = """
    .gradio-container { background: #0f0f1a !important; }
    .gr-button-primary { background: #2ecc71 !important; border-color: #27ae60 !important; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="🌿 Détection Maladies des Plantes"
    ) as demo:

        gr.HTML("""
        <div style="text-align:center; padding:20px; background:linear-gradient(135deg,#0d2b1a,#1a3a2a);
                    border-radius:12px; margin-bottom:10px;">
            <h1 style="color:#2ecc71; font-size:2em; margin:0;">🌿 Plant Disease Detector</h1>
            <p style="color:#aaa; margin:8px 0 0;">
                Détection automatique de maladies sur feuilles de tomate
                — 4 classes · PlantVillage Dataset
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 Uploader une image de feuille",
                    type="pil",
                    height=280
                )
                model_choice = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0],
                    label="🤖 Choisir le modèle"
                )
                predict_btn = gr.Button("🔍 Analyser", variant="primary", size="lg")

                gr.Examples(
                    examples=[],  # Ajouter des images exemple si disponibles
                    inputs=image_input,
                    label="Exemples"
                )

            with gr.Column(scale=2):
                result_output = gr.HTML(label="Résultat")
                proba_output  = gr.HTML(label="Probabilités")

        with gr.Accordion("🔬 Visualisation du Prétraitement", open=False):
            preprocess_output = gr.Image(label="Étapes de prétraitement appliquées", height=380)

        predict_btn.click(
            fn=predict,
            inputs=[image_input, model_choice],
            outputs=[result_output, proba_output, preprocess_output]
        )

        gr.HTML("""
        <div style="text-align:center; padding:15px; color:#555; font-size:0.85em; margin-top:10px;">
            Projet Traitement d'Image · PlantVillage Dataset ·
            Random Forest | SVM | KNN | CNN
        </div>
        """)

    return demo, css

# ─── Lancement ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🌿 Lancement de l'interface Gradio...")
    print("📌 Ouvre ton navigateur sur : http://127.0.0.1:7860\n")
    demo, css = build_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,       # Mettre True pour partager via URL publique
        inbrowser=True,    # Ouvre automatiquement le navigateur
        theme=gr.themes.Base(
            primary_hue="green",
            neutral_hue="slate",
        ),
        css=css
    )