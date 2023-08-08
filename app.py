import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import streamlit as st
import subprocess
import csv
import joblib
from pypdb import *
from sklearn.preprocessing import StandardScaler
import py3Dmol
from stmol import showmol


@st.cache_data
def calculate_lipinski_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("You entered an invalid SMILES string")

    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'CarbonCount': Descriptors.HeavyAtomCount(mol),
        'OxygenCount': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    }

    return pd.DataFrame(descriptors, index=[0])


def generate_csv_file(string1, string2, filename):
    data = [[string1 + '\t' + string2]]  # Create a list of lists containing the strings
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

@st.cache_data
def search_pdb(smiles):
    result = Query(smiles).search()
    prot_str = ""
    for i, pdb_id in enumerate(result):
        if i < 30:
            prot_str += pdb_id + ","
        else:
            break
    prot_str = prot_str.rstrip(",")  # Remove the trailing comma and space
    return prot_str


def prot_visualization(prot_str):
    # The context manager for the sidebar
    with st.sidebar:
        st.subheader('Customize Visualization Menu')
        prot_list = prot_str.split(',')
        bcolor = st.color_picker('Pick A Color', '#0046F9')
        protein = st.selectbox('select protein', prot_list)
        style = st.selectbox('style', ['cartoon', 'cross', 'stick', 'sphere', 'line', 'sphere'])
        spin = st.checkbox('Spin', value=False)

    # Create the 3D molecular view outside the context manager to keep it visible
    xyzview = py3Dmol.view(query='pdb:' + protein)

    # Check if a valid PDB ID was retrieved
    if protein.strip():
        xyzview.setStyle({style: {'color': 'spectrum'}})
        xyzview.setBackgroundColor(bcolor)
        if spin:
            xyzview.spin(True)
        else:
            xyzview.spin(False)
        xyzview.zoomTo()
        showmol(xyzview, height=500, width=800)
    else:
        st.warning("No valid PDB ID found for the given compound.")


# Set page title and initial layout
st.set_page_config(page_title="Plasmodium Drug Predictor", page_icon='🦟', layout="wide")

# Apply custom CSS to modify app appearance


# App title
st.title("Antiplasmodium Drug Prediction Platform")

# Description
description = """
The "Antiplasmodium Drug Prediction Platform" is a web application designed to aid in the prediction and analysis of potential antiplasmodium drug compounds. It offers several key features, including the calculation of Lipinski's descriptors to assess a compound's drug-likeness, prediction of a compound's activity and pIC50 value, and visualization of interacting proteins using SMILES input. The app leverages various libraries such as RDKit, Streamlit, Py3Dmol, and scikit-learn to provide an interactive and informative environment for researchers and scientists working in the field of antiplasmodium drug discovery.
"""

# Display the description
st.markdown(description)


# ---Use local CSS---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style/style.css")


# Main function
def main():
    """
    Main function for the Plasmodium faliparum Drug Prediction System web application.
    """
    # Text input
    smiles_input = st.text_input("Enter Canonical SMILES")

    # Selectbox options
    options = [
        "Compute Lipinski's Descriptors",
        "Predict the Compound's Activity",
        "Predict the Compound's pIC50",
        #"Retrieve interacting proteins"
    ]

    # Selectbox
    selected_option = st.selectbox("Select the program to run", options)

    # Button to run the selected option
    if st.button("Run"):
        if not smiles_input:
            st.text("You did not Enter Canonical SMILES")
        else:
            try:
                if selected_option == "Compute Lipinski's Descriptors":
                    # Calculate Lipinski's descriptors and display the result
                    df = calculate_lipinski_descriptors(smiles_input)
                    df.columns = ['Molecular Weight', 'Octanol-Water Partition Coefficient (LogP)',
                                  'Number of Hydrogen Bond Donors', 'Number of Hydrogen Bond Acceptors',
                                  'Number of Rotatable Hydrogen Bonds', 'Carbon atom Count', 'Oxygen Atom Count']
                    hide_table_row_index = """
                        <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                        </style>
                    """
                    st.markdown(hide_table_row_index, unsafe_allow_html=True)
                    st.table(df)
                elif selected_option == "Predict the Compound's Activity":
                    X = calculate_lipinski_descriptors(smiles_input)
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                    # Load the model and predict the compound's activity
                    loaded_model = joblib.load('MLPC_2_lipinsky_model.joblib')
                    y_pred = loaded_model.predict(X)

                    if y_pred == [1]:
                        st.text('Active')
                    else:
                        st.text('Inactive')
                elif selected_option == "Predict the Compound's pIC50":
                    # Generate a CSV file and compute pIC50
                    string1 = smiles_input
                    string2 = 'Compound_name'

                    filename = "molecule.smi"
                    generate_csv_file(string1, string2, filename)

                    script_path = "./padel.sh"
                    subprocess.call(['bash', script_path])

                    data = pd.read_csv('descriptors_output.csv')

                    X = data.drop(columns=['Name'])

                    loaded_model = joblib.load('padel_model.joblib')
                    y_pred = loaded_model.predict(X)
                    predicted_value = y_pred[0]
                    predicted_value = format(predicted_value, ".2f")
                    st.text("The pIC50 of your compound is " + str(predicted_value))

            except ValueError as e:
                st.error(str(e))

    with st.container():
        st.write("---")
        st.subheader('Interacting Proteins')

        try:
            mol = Chem.MolFromSmiles(smiles_input)

            if mol is None:
                st.text("There are no proteins to view. Please enter valid canonical SMILES")
            else:
                if smiles_input:
                    prot_visualization(search_pdb(smiles_input))
                else:
                    st.text("There are no proteins to view. You did not enter canonical SMILES.")
        except Exception as e:
            st.text("Error: " + str(e))

if __name__ == "__main__":
    main()

st.write('##')
st.write('___')
text = "**Authors**<br>1. Edwin Mwakio, Masinde Muliro University of Science and Technology<br>2. Clabe Wekesa, [International Max Planck Institute for Chemical Ecology](https://www.ice.mpg.de/person/128684/2824)<br>3. Patrick Okoth, [Masinde Muliro University of Science and Technology](https://mmust.ac.ke/staffprofiles/index.php/dr-patrick-okoth)"

st.markdown(text, unsafe_allow_html=True)
st.write('___')