import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx


# Filter Data set 3 into a usable txt file first
input_file = open("41467_2014_BFncomms5212_MOESM1045_ESM.txt", "r")
output_file = open("filtered_data.txt", "w")

# Define the chunk size
chunk_size = 10000

# Initialize an empty list to store filtered data
filtered_data = []

# Read the file in chunks and filter
for chunk in pd.read_csv(input_file, sep='\t', chunksize=chunk_size):
    # Filter the chunk based on "PubMed occurrence" column
    filtered_chunk = chunk[chunk['PubMed occurrence'] > 500]
    # Append the filtered chunk to the list
    filtered_data.append(filtered_chunk)

# Concatenate all the filtered chunks
filtered_df = pd.concat(filtered_data)

# Write the filtered data to a new file
filtered_df.to_csv(output_file, sep='\t', index=False)

# Load the data
symptom_data_path = open("41467_2014_BFncomms5212_MOESM1044_ESM.txt", "r")
disease_data_path = open("filtered_data.txt", "r")

# Read the symptom data
symptom_data = pd.read_csv(symptom_data_path, sep='\t')
# Read the disease data
disease_data = pd.read_csv(disease_data_path, sep='\t')

# Filter instances with occurrences greater than 500
filtered_symptom_data = symptom_data[symptom_data['PubMed occurrence'] > 500]
filtered_disease_data = disease_data[disease_data['PubMed occurrence'] > 500]

# Initialize the Bayesian Network
model = BayesianNetwork()

# Add nodes (diseases and symptoms)
disease_nodes = filtered_disease_data['MeSH Disease Term'].tolist()
symptom_nodes = filtered_symptom_data['MeSH Symptom Term'].tolist()


# Function to check if adding an edge creates a cycle
def check_for_cycle(model, u, v):
    temp_model = model.copy()
    temp_model.add_edge(u, v)
    return not nx.is_directed_acyclic_graph(nx.DiGraph(temp_model.edges()))


# Add edges (disease -> symptom relationship) without self-loops and cycles
for disease in disease_nodes:
    for symptom in symptom_nodes:
        if disease != symptom and check_for_cycle(model, disease, symptom):
            model.add_edge(disease, symptom)

# Initialize priors using disease occurrence data
cpds = []
for disease in disease_nodes:
    disease_occurrence = filtered_disease_data[filtered_disease_data['MeSH Disease Term'] == disease]['PubMed occurrence'].values[0]
    total_occurrences = filtered_disease_data['PubMed occurrence'].sum()
    disease_prob = disease_occurrence / total_occurrences
    cpd = TabularCPD(variable=disease, variable_card=2, values=[[1 - disease_prob], [disease_prob]])
    cpds.append(cpd)

# Calculate conditional probability tables using symptom data
for symptom in symptom_nodes:
    symptom_occurrences = \
    filtered_symptom_data[filtered_symptom_data['MeSH Symptom Term'] == symptom]['PubMed occurrence'].values[0]
    total_occurrences = filtered_symptom_data['PubMed occurrence'].sum()
    idiopathic_prob = symptom_occurrences / total_occurrences

    for disease in disease_nodes:
        if model.has_edge(disease, symptom):
            # Example CPT: this should be based on actual co-occurrence data
            # For simplicity, let's assume a random probability
            cpd = TabularCPD(
                variable=symptom,
                variable_card=2,
                values=[[0.95, 0.8], [0.05, 0.2]],
                evidence=[disease],
                evidence_card=[2]
            )
            cpds.append(cpd)

    # Add an idiopathic disease to account for symptoms without known causes
    if not any(disease in model.get_parents(symptom) for disease in disease_nodes):
        cpd_idiopathic = TabularCPD(
            variable=symptom,
            variable_card=2,
            values=[[1 - idiopathic_prob, 1 - idiopathic_prob], [idiopathic_prob, idiopathic_prob]],
            evidence=['Idiopathic'],
            evidence_card=[2]
        )
        cpds.append(cpd_idiopathic)

# Add CPDs to the model
model.add_cpds(*cpds)

# Verify the model
assert model.check_model(), "The model structure is incorrect"

# Perform inference
inference = VariableElimination(model)


# Step 6: Create an interface for symptom input and disease prediction
def predict_disease(symptoms):
    evidence = {symptom: 1 for symptom in symptoms}
    results = {}
    for disease in disease_nodes:
        result = inference.query(variables=[disease], evidence=evidence)
        results[disease] = result.values[1]
    most_likely_disease = max(results, key=results.get)
    return most_likely_disease, results


