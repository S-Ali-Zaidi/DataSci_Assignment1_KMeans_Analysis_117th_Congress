#### Data Science Fundamentals: Assignment 1
## The 117th Congress: Assessing Congressional Committee Referrals Using NLP Embeddings and Unsupervised Clustering of Legislative Texts

> [!NOTE] 
>**K-Means Clustering UMAP:** High-Dimensional NLP Embeddings of 11,154 House bills introduced in the 117th Congress, labeled by semantic theme per Cluster

![UMAP Projection of Bill Embeddings](https://raw.githubusercontent.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/refs/heads/main/Data/Main_Data/Results/UMAP_Projection_Cluster_Labeled_Seed7_N1000_Inertia4667.89.png)

### Motivation

Having served as a congressional staffer in the U.S. House of Representatives, I have firsthand experience with the complexities and procedural bottlenecks of the legislative process. For instance, during the 117th Congress (which was in session from January 2021 to January 2023), over **11,460 bills and resolutions** were introduced across 330 legislative days—averaging about **34 new pieces of legislation per day**.

Each bill requires referral to one or more of the 21 standing or select committees, a task that is both labor-intensive and time-sensitive. This volume of legislation can place a significant burden on the House Speaker, House Parliamentarian, and their respective staff to assign bills appropriately, relying not only on intricate knowledge of committee jurisdictions, but also factoring in legislative and political nuances.

Recognizing the potential for advances in Transformer-based Neural Networks to assist in the streamlining of such processes—especially for less critical legislation and procedures—I was motivated to explore whether **natural language processing (NLP)** techniques and **unsupervised learning** algorithms could inform how we might automate, assist, or assess the committee referral process. 

Specifically, I aimed to investigate if clustering bill texts—based on semantic similarities identified within their embedded representations—could reveal thematic structures aligning with committee jurisdictions, thereby serving as a foundation for an automated referral system.

## Process

> [!TIP]
> - **Scripts:** 15 different scripts Python scripts were written for each part of the process, all of which are linked below or can be found in the rep [here](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/tree/main/Scripts). Each Python script is annotated with my comments explaining certain aspects of the process in more detail.
>
> - **Data**: All data, including the [LegiScan `.json` files](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/tree/main/Data/US%203/2021-2022_117th_Congress/bill), [original `.pdf` files, cleaned `.txt files`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/tree/main/Data/bill_texts), and [`.npy` embedding files](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/tree/main/Data/embeddings) for all [**11,154** House bills](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Data/Main_Data/Bills.csv) and [21 Committees](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Data/Main_Data/Committees.csv) I analyzed are in the repo and available for reproduction or further analysis.  
>
> - **Results**: Including the Heat-map and UMAP plots, the `.npy` array for the K-Means centroids, individual distance and silhouette metrics for each bill, and a curated sample of bills from each cluster can all be found [here](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/tree/main/Data/Main_Data/Results).

Following the **CRISP-DM** ([Cross-Industry Standard Process for Data Mining](https://www.datascience-pm.com/crisp-dm-2/)) methodology, I structured the project into the following phases: 

### 1. Domain Understanding

The primary objective was to determine if NLP models and clustering algorithms could effectively predict House committee assignments, potentially easing the workload of the House Parliamentarian and expediting the legislative process. My hypothesis was that by clustering bills based on their semantic content, we might uncover patterns that mirror the existing committee referral system.

### 2. Data Understanding
**Scripts Used:** [`bill_pull.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/bill_pull.py), [`bill_titles.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/bill_titles.py)

To have a complete dataset, I focused on the recent 117th Congress, rather than the current 118th. I sourced comprehensive legislative data from **LegiScan**, which provided detailed metadata for each bill, including bill numbers, titles, sponsors, initial committee assignments, and links to the full texts on Congress.gov. I focused only on House Legislation to avoid the complexities of dealing with the alternative Committee structure within the Senate. (I'm also partial to the House, having worked only for Members of that Chamber).

- **Metadata Parsing:** Using `bill_pull.py` and `bill_titles.py`, I extracted essential metadata and ensured completeness. After parsing and cleaning, I had a dataset of **11,154 bills**—about **97%** of all bills introduced during that Congress.

### 3. Data Preparation

**Scripts Used:** [`bill_download.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/bill_cleaner_pt1.py),  [`bill_convert_TXT.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/bill_convert_TXT.py), [`data_check.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/data_check.py), [`bill_cleaner_part1.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/bill_cleaner_pt1.py), [`bill_cleaner_part2.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/bill_cleaner_pt2.py), [`bill_counter.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/bill_counter.py), [`committee_counter.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/committee_counter.py), [`embedding_committee_rules.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/embedding_committee_rules.py), [`embedding_bills_under_8100.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/embedding_bills_under_8100.py), [`embedding_bills_over_8100.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/embedding_bills_over_8100.py)

1. **Data Extraction and Cleaning:**
   - **PDF Downloading and Conversion:** With `bill_download.py` and `bill_convert_text.py`, I data-mined all bill PDFs from Congress.gov (over 2GB worth of PDF Data) and converted them into plain text files.
   - **Integrity Check:** I regularly employed `data_check.py` to ensure that each bill in my master CSV file had corresponding PDF and text files, flagging any discrepancies.

2. **Text Cleaning:**
   - **Removing Extraneous Information:** I used `bill_cleaner_part1.py` and `bill_cleaner_part2.py` to strip out unnecessary metadata, such as sponsor names and prior committee assignments, to prevent bias in the embeddings influencing the clustering and later prediction process. 
   - **Standardizing Formats:** I ensured consistency across all bill texts to focus on the legislative content through a manual and programatic analysis of a sample of the cleaned text files, adjusting both `bill_cleaner_part1.py` and `bill_cleaner_part2.py` until I was satisfied that all final text files had parity for the embedding process. 

3. **Committee Representation:**
   - **Compiling Descriptions:** I extracted textual descriptions of each of the **21 House committees** from [the House Rules for the 117th Congress](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Data/US%203/117th%20House%20Rules.md), which outlined their responsibilities, powers, and areas of jurisdiction. These 21 text documents were augmented with detailed biographical information authored by the committees themselves, published on each of the  official committee websites.
   - **Creating Representative Texts:** These descriptions served as representative documents for each committees, which could be compared in the embedding space shared with the legislative texts. 

4. **Model Selection and Pre-Embedding Preparation:**
   - **NLP Model**: [OpenAI's `text-embedding-3-large`](https://platform.openai.com/docs/guides/embeddings/) was selected to embed all texts, due to its large token window capacity, my familiarity with OpenAI API, and the preferable cost-to-quality of OpenAI models. Embedding all legislation and committee texts cost less than $7 total in API credit. Additionally, OpenAI embeddings pre-normalized to the unit norm, allowing me to skip a step to normalize vectors to account for magnitude. 
   - **Token Limitations:** Due to `text-embedding-3-large`maximum token limit of **8,190 tokens**, I used `bill_counter.py` and `committee_counter.py` to identify texts exceeding this limit. No committee texts were above the limit, while around 600 legislative texts were above the limit. 

5. **Generating Embeddings:**
   - **Bills:** I employed `embedding_bills_under_8100.py` for bills within the token limit.
   - **Chunking Long Texts:** For bills over the limit, `embedding_bills_over_8100.py` split them into manageable chunks, embedded each, and averaged the embeddings to represent the entire bill. 
   - **Committees:** I used `embedding_committee_rules.py` to create embeddings for each of the 21 committee's textual representation.

### 4. Modeling

**Script Used:** [`kmeans_predict_rules.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/kmeans_predict_rules.py)

- **Clustering Approach:** I implemented **K-Means clustering** with **k=21**, matching the number of House committees, hypothesizing that a cluster may form representing each Committee's area of jurisdiction.
- **Hyper-parameters:** I tested multiple random seeds and increased the number of initializations for each up to 5000, to ensure convergence and reproducibility. All seeds converged to the same approximate centroids within 1000 initializations. 
- **Visual Graphs**: I chose to rely on the use of heat-map plots that illustrated the number of bills per committee that fell into each cluster, as well as UMAPs of the clustering in 2d, colored by either committee labels or cluster labels.
- **Assignment Prediction:** I mapped committee embeddings to clusters within the heat-map graphic to assess alignment of each cluster with with their respective bills.

### 5. Evaluation

**Scripts Used:** [`explore_clusters.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/explore_clusters.py), [`ClusterGraphic.py`](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Scripts/ClusterGraphic.py)

- **Cluster Analysis:**
  - After identifying widespread anomalies within the graphical results, I used `explore_clusters.py` to extract examine representative bills in each cluster (by proximity to the centroid), identifying common themes within the cluster. 
  - Many clusters were identified not to correspond to specific jurisdictions, but themes that were sometimes more abstract or more specific. These labels were documented and incorporated in further scripting. 
- **Visualization:**
  - Finally I generated the final heat-maps and UMAP projections with `ClusterGraphic.py` to visualize relationships between clusters, bills, committees, and the cluster themes.
  - The final heat-maps illustrated how many bills from each committee fell into each cluster, revealing patterns and anomalies that made sense only in light of a deeper textual examination. 

## Results

The clustering yielded some insightful, albeit unexpected, results:

- **Partial Alignment:**
  - Certain clusters closely aligned with specific committees.
    - **Cluster 3** was dominated by bills related to veterans' services and benefits, aligning with the **Veterans Affairs** and **Armed Services** committees. Both the bills and the committee embeddings fell into this cluster.
    - **Cluster 12** centered on foreign policy and national security, matching the **Foreign Affairs** and **Intelligence** committees.

- **Discrepancies:**
  - In many cases, committee embeddings did not fall into clusters containing the majority of their bills.
    - For example, **Energy and Commerce** had its highest number of bills in **Cluster 8**, yet its committee embedding fell into **Cluster 11**.
  - Some clusters were thematic but spanned multiple committees, indicating overlapping jurisdictions.

> [!NOTE]
> **Heatmap: Committee Predictions are Noted by Cells Outlined in Blue**

![K-Means Heatmap: Bills Per Committee per Cluster](https://raw.githubusercontent.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/refs/heads/main/Data/Main_Data/Results/KMeans_Heatmap_Seed7_N1000_Inertia4667.89.png)

Below is a table identifying the major themes of each cluster. To see examples of specific bills that fell into each cluster, and helped shape these themes, [go here.](https://github.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/blob/main/Data/Main_Data/Results/Clusters.md)

| **Cluster Number** | **Identified Theme**                                                                      | **Committees with Strongest Bill Showings**                       | **Committees with Embeddings in Cluster**                                   |
| ------------------ | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **0**              | Prohibition of Federal Funding for COVID-19 Mandates and Limitation of Government Actions | Homeland Security, Judiciary, Oversight and Reform                | None                                                                        |
| **1**              | Education and Workforce Development for Underserved Communities                           | Education and Labor                                               | None                                                                        |
| **2**              | Congressional Oversight and Requests for Executive Documentation                          | Rules, Judiciary, Oversight and Reform                            | Rules, Budget, Ethics, Oversight and Reform, Judiciary, Admin               |
| **3**              | Enhancement of Veterans’ Services and Benefits                                            | Veterans' Affairs, Armed Services                                 | Veterans' Affairs, Armed Services                                           |
| **4**              | Environmental Conservation and Renewable Energy Initiatives                               | Natural Resources, Transportation, Energy & Commerce              | None                                                                        |
| **5**              | Healthcare Access Improvement and Medicare Policy Reforms                                 | Energy and Commerce, Ways and Means                               | None                                                                        |
| **6**              | Recognition and Commemoration of Military and Service Personnel                           | Armed Services, Financial Services                                | None                                                                        |
| **7**              | Government Ethics, Accountability, and Regulatory Reforms                                 | Judiciary, Energy and Commerce, Financial Services                | None                                                                        |
| **8**              | Healthcare Access and Public Health Reforms                                               | Energy and Commerce                                               | None                                                                        |
| **9**              | Designation of National Awareness Weeks and Observances                                   | Education and Labor, Energy and Commerce, Oversight               | None                                                                        |
| **10**             | Tax Policy Amendments and Economic Incentives                                             | Ways and Means                                                    | Ways and Means                                                              |
| **11**             | Infrastructure Development and Environmental Management                                   | Natural Resources, Transportation, Energy & Commerce, Agriculture | Science, Space & Tech, Energy & Commerce, Transportation, Natural Resources |
| **12**             | Foreign Policy, Sanctions, and National Security                                          | Foreign Affairs, Intelligence                                     | Foreign Affairs, Intelligence                                               |
| **13**             | Law Enforcement, Immigration, and Border Security Policies                                | Judiciary, Homeland Security                                      | Homeland Security                                                           |
| **14**             | Tax Policy Amendments, Economic Incentives, and Infrastructure Investment                 | Ways and Means                                                    | None                                                                        |
| **15**             | Comprehensive Infrastructure, Climate Action, Health, Housing, and Democracy Protection   | Energy and Commerce, Financial Services, Natural Resources        | None                                                                        |
| **16**             | Gun Control and Firearms Regulation                                                       | Judiciary                                                         | None                                                                        |
| **17**             | Education, Small Business Support, Social Programs, and Regulatory Reforms                | Education and Labor, Financial Services, Small Business           | Education and Labor, Financial Services, Small Business                     |
| **18**             | Resolutions and Recognitions for Social Issues, Human Rights, and Commendations           | Foreign Affairs, Oversight                                        | None                                                                        |
| **19**             | Law Enforcement Recognition, National Security, and Social Resolutions                    | Foreign Affairs, Judiciary, Oversight                             | None                                                                        |
| **20**             | Designation of Federal and Postal Facilities and Memorials                                | Oversight and Reform                                              | None                                                                        |

> [!NOTE]
> **How Each Cluster Relates to Others Clusters**

![Cluster Distance Heatmap](https://raw.githubusercontent.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/refs/heads/main/Data/Main_Data/Results/Cluster_Distance_Heatmap_Seed7_N1000_Inertia4667.89.png)

## Difficulties

The project presented several challenges:

1. **Data Acquisition and Processing:**
   - **Text Availability:** Obtaining plain text versions of all bills was more challenging than I anticipated, requiring the process of downloading and conversion of over 2gb of PDF files.
   - **Data Cleaning:** The presence of hidden metadata within the PDF files that appeared in the TXT files -- and the somewhat inconsistent formatting between different bill types -- necessitated extensive cleaning to ensure accurate embeddings.

2. **Token Limitations:**
   - **Embedding Constraints:** Handling bills exceeding the 8,190-token limit of the embedding model required strategic chunking, embedding, and averaging. This required a standalone script, and added some complexity to the process.

3. **Clustering Complexities:**
   - **Semantic Overlap:** The high dimensionality and semantic overlap of legislative texts made it difficult for clusters to neatly correspond to committee jurisdictions, going against my hypothesis.
   - **Discrepancies:** Committee embeddings sometimes did not align with clusters containing the majority of their bills, suggesting that some textual descriptions I compiled may not fully capture the full scope of certain committee's activities and areas of jurisdiction.

4. **Unexpected Results:**
   - **Thematic Broadness:** Some clusters encompassed broad themes spanning multiple committees, reflecting the interconnected nature of legislative issues.
   - **Overlapping Jurisdictions:** Highlighted the complexity of congressional committees where subject matters often intersect, and legislation often passes through multiple Committees beyond the Committee they are initially referred to. 
   - **Deeper Textual Analysis:** Visual results depicted in the heat-map and UMAP plots caused more confusion than clarity initially. Only after a deeper analysis of representative texts, and labeling of the plots by the identified themes, did the visual results begin to make more sense. 

## Conclusion

While the clustering approach revealed interesting and coherent thematic groupings / underlying structures within the legislative texts, it did not consistently predict or correlate to committee assignments of these texts. This suggests that although NLP and unsupervised learning hold potential for augmenting the legislative process, further refinement is necessary.

**Future Directions:**

- **Supervised Learning Integration:** Training classifier or LLM models on existing committee assignments to enhance prediction capabilities.
- **Expanded Data Features:** Expanded texts for Committee embeddings, as well as exploring ways to include metadata within legislative text representations, such as bill sponsors, political context, and historical referral patterns.
- **Alternative Clustering Methods:** Exploring algorithms like DBSCAN or hierarchical clustering to capture more complex relationships.

> [!NOTE]
> **UMAP of Bills labeled by Committee Assignments**

![UMAP: Committee Assignments](https://raw.githubusercontent.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/refs/heads/main/Data/Main_Data/Results/UMAP_Projection_Committee_Seed7_N1000_Inertia4667.89.png)

> [!NOTE]
> **UMAP of Bills labeled by Cluster Theme**

![UMAP: Projection of Bill Embeddings](https://raw.githubusercontent.com/S-Ali-Zaidi/DataSci_Assignment1_KMeans_Analysis_117th_Congress/refs/heads/main/Data/Main_Data/Results/UMAP_Projection_Cluster_Labeled_Seed7_N1000_Inertia4667.89.png)

