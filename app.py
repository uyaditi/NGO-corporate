from flask import Flask, request, jsonify
import numpy as np
from scipy.optimize import linear_sum_assignment

app = Flask(__name__)


def calculate_match_score(ngo, corp):
    """
    Calculate match score between an NGO and a Corporate.
    """
    # **Sector Match Score** (Jaccard Similarity)
    ngo_sectors = set(ngo.get("sector", "").lower().split(", "))
    corp_impact_areas = set(corp.get("impact_areas", "").lower().split(", "))
    sector_match = len(ngo_sectors & corp_impact_areas) / \
        max(len(ngo_sectors | corp_impact_areas), 1)

    # **Budget Suitability** (CSR Budget vs NGO Needs)
    budget_score = min(corp["csr_budget"] /
                       max(ngo["needs_budget"], 1), 1.0)  # Normalized

    # **Location Proximity Score** (1 = same city, 0.5 = same state, 0 otherwise)
    ngo_location = (ngo.get("city", ""), ngo.get(
        "state", ""), ngo.get("country", ""))
    corp_location = (corp.get("city", ""), corp.get(
        "state", ""), corp.get("country", ""))
    geo_score = 1 if ngo_location[0] == corp_location[0] else (
        0.5 if ngo_location[1] == corp_location[1] else 0)

    # Final weighted score
    return (0.5 * sector_match) + (0.3 * budget_score) + (0.2 * geo_score)


def build_cost_matrix(ngos, corporates):
    """
    Create a cost matrix (negative scores for Hungarian algorithm).
    """
    cost_matrix = np.zeros((len(ngos), len(corporates)))

    for i, ngo in enumerate(ngos):
        for j, corp in enumerate(corporates):
            # Negative because Hungarian finds min cost
            cost_matrix[i][j] = -calculate_match_score(ngo, corp)

    return cost_matrix


@app.route('/match/optimal', methods=['POST'])
def optimal_matching():
    """
    API Endpoint: Return optimal NGO-Corporate pairings using Hungarian Algorithm.
    """
    data = request.json
    ngos = data.get("ngos", [])
    corporates = data.get("corporates", [])

    if not ngos or not corporates:
        return jsonify({"error": "Invalid input. Provide 'ngos' and 'corporates' list"}), 400

    cost_matrix = build_cost_matrix(ngos, corporates)
    ngo_indices, corp_indices = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    for i, j in zip(ngo_indices, corp_indices):
        matched_pairs.append({
            "ngo_id": ngos[i]["id"],
            "ngo_name": ngos[i]["name"],
            "corporate_id": corporates[j]["id"],
            "corporate_name": corporates[j]["name"],
            "match_score": -cost_matrix[i][j]  # Convert back to positive
        })

    return jsonify({"matched_pairs": matched_pairs})


@app.route('/match/scores', methods=['POST'])
def get_match_scores():
    """
    API Endpoint: Return all match scores between NGOs and Corporates.
    """
    data = request.json
    ngos = data.get("ngos", [])
    corporates = data.get("corporates", [])

    if not ngos or not corporates:
        return jsonify({"error": "Invalid input. Provide 'ngos' and 'corporates' list"}), 400

    scores = [
        {"ngo_id": ngo["id"], "ngo_name": ngo["name"],
         "corporate_id": corp["id"], "corporate_name": corp["name"],
         "match_score": calculate_match_score(ngo, corp)}
        for ngo in ngos for corp in corporates
    ]

    return jsonify({"match_scores": scores})


if __name__ == '__main__':
    app.run(debug=True)
