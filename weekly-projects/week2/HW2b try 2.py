import pandas as pd
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
import networkx as nx


# HW2 Part B Step 3
# Filter Data Set 3 into a usable txt file first
input_file = open("41467_2014_BFncomms5212_MOESM1045_ESM.txt", "r")
output_file = open("filtered_data.txt", "w")

# Define the chunk size
chunk_size = 10000

# Initialize an empty list to store filtered data
filtered_data = []

# Read the file in chunks and filter
for chunk in pd.read_csv(input_file, sep='\t', chunksize=chunk_size):
    # Filter the chunk based on "PubMed occurrence" column
    filtered_chunk = chunk[chunk['PubMed occurrence'] > 300]
    # Append the filtered chunk to the list
    filtered_data.append(filtered_chunk)

# Concatenate all the filtered chunks
filtered_df = pd.concat(filtered_data)

# Write the filtered data to a new file
filtered_df.to_csv(output_file, sep='\t', index=False)

# Now create the Bayesian network
# Read the data from the file
data = pd.read_csv('filtered_data.txt', sep='\t')

# Reclassify diseases and symptoms
data['Disease Node'] = data['MeSH Disease Term'] + ' (disease)'
data['Symptom Node'] = data['MeSH Symptom Term'] + ' (symptom)'

# Create a Bayesian Network
model = BayesianNetwork()

# Add nodes for diseases and symptoms
diseases = data['Disease Node'].unique()
symptoms = data['Symptom Node'].unique()

nodes = list(diseases) + list(symptoms)
model.add_nodes_from(nodes)

# Add edges based on interactions
edges = [(row['Disease Node'], row['Symptom Node']) for index, row in data.iterrows()]
model.add_edges_from(edges)

# Print the model
print("Nodes: ", model.nodes())
print("Edges: ", model.edges())

# HW2 Part B Step 4
# Read the disease occurrence data
occurrence_data = pd.read_csv('41467_2014_BFncomms5212_MOESM1043_ESM.txt', sep='\t')

# Reclassify the disease names to match the Bayesian Network
occurrence_data['Disease Node'] = occurrence_data['MeSH Disease Term'] + ' (disease)'

# Reclassify the disease names to match the Bayesian Network
occurrence_data['Disease Node'] = occurrence_data['MeSH Disease Term'] + ' (disease)'

# Filter the diseases that are present in the Bayesian Network
relevant_occurrences = occurrence_data[occurrence_data['Disease Node'].isin(diseases)].copy()

# Calculate the prior probabilities for each disease
total_occurrences = relevant_occurrences['PubMed occurrence'].sum()
relevant_occurrences.loc[:, 'Prior Probability'] = relevant_occurrences['PubMed occurrence'] / total_occurrences

# Define the CPDs for each disease node based on the prior probabilities
cpds = []
for _, row in relevant_occurrences.iterrows():
    cpd = TabularCPD(variable=row['Disease Node'], variable_card=2, values=[[1 - row['Prior Probability']],
                                                                            [row['Prior Probability']]])
    cpds.append(cpd)

# Ensure all symptom nodes have CPDs
existing_cpds = model.get_cpds()

# Create a set of nodes that have CPDs
nodes_with_cpds = {cpd.variable for cpd in existing_cpds}

# Add CPDs for symptom nodes that don't have one
for symptom in symptoms:
    if symptom not in nodes_with_cpds:
        # Assign a default CPD for the symptom
        # Assuming equal probability of having or not having the symptom
        cpd = TabularCPD(variable=symptom, variable_card=2, values=[[0.5], [0.5]])
        model.add_cpds(cpd)

# HW2 Part B Step 5
# Calculate co-occurrence probabilities for each disease-symptom pair
# for disease in diseases:
#     for symptom in symptoms:
#         co_occurrences = data[(data['Disease Node'] == disease) & (data['Symptom Node'] == symptom)]
#         if not co_occurrences.empty:
#             # Calculate the probabilities
#             symptom_given_disease = co_occurrences['PubMed occurrence'].sum() / relevant_occurrences.loc[
#                 relevant_occurrences['Disease Node'] == disease, 'PubMed occurrence'].values[0]
#             symptom_given_no_disease = (occurrence_data.loc[
#                                             occurrence_data['Disease Node'] != disease, 'PubMed occurrence'].sum() -
#                                         co_occurrences['PubMed occurrence'].sum()) / total_occurrences
#
#             # Create CPT for the symptom node
#             cpd = TabularCPD(variable=symptom, variable_card=2,
#                              values=[[1 - symptom_given_disease, 1 - symptom_given_no_disease],
#                                      [symptom_given_disease, symptom_given_no_disease]],
#                              evidence=[disease], evidence_card=[2])
#             cpds.append(cpd)

# Add the CPDs to the Bayesian Network
model.add_cpds(*cpds)

# Verify the model
try:
    model.check_model()
    print("Model is valid.")
except ValueError as e:
    print(f"Model validation error: {e}")

print(cpds)

# HW2 Part B Step 6
def infer_disease(model, observed_symptoms):
    # Initialize the inference object
    inference = VariableElimination(model)

    # Prepare the evidence
    evidence = {symptom: 1 for symptom in observed_symptoms}

    # Query the network to find the most probable disease
    diseases = [node for node in model.nodes() if node.endswith('(disease)')]
    result = inference.map_query(variables=diseases, evidence=evidence)

    return result


def main():
    # Get the list of symptoms from the user
    observed_symptoms = input("Enter the observed symptoms, separated by commas: ").split(", ")
    observed_symptoms = [symptom.strip() + " (symptom)" for symptom in observed_symptoms]

    # Infer the most likely disease
    most_likely_disease = infer_disease(model, observed_symptoms)

    # Print the result
    print("The most likely disease based on the observed symptoms is:")
    for disease, value in most_likely_disease.items():
        if value == 1:
            print(disease)


if __name__ == "__main__":
    main()
