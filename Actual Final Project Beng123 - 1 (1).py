## This is a sample code from one of my school projects. Here the ultimate goal was to map all the cellular pathways that the body's cells goes through.
## By creating this pathway, we were able to replicate a healthy Red blood cell as our control, then insert a disease model Red blood cells.
# Based upon the graphs and stuff, it will help us suggest possible future treatments for physicians to suggest.
# 
# **Reactions:**
# 
# 1. G6PDH2r_1: g6pdh2r_c + nadph_c <=> g6pdh2r_nadph_c
# 2. G6PDH2r_2: atp_c + g6pdh2r_c <=> g6pdh2r_atp_c
# 3. G6PDH2r_3: g6p_c + g6pdh2r_c <=> g6pdh2r_g6p_c
# 4. G6PDH2r_4: g6pdh2r_6pgl_c <=> _6pgl_c + g6pdh2r_c
# 5. G6PDH2r_5: g6pdh2r_g6p_c + nadp_c <=> g6pdh2r_g6p_nadp_c
# 6. G6PDH2r_6: g6pdh2r_6pgl_nadph_c <=> g6pdh2r_6pgl_c + nadph_c
# 7. G6PDH2r_7: g6pdh2r_g6p_nadp_c <=> g6pdh2r_6pgl_nadph_c + h_c
# 
# 
# **Dissociation Constants:** 
# Experimental data gives the following for the dissociation constants: 
# 
# $$K_{i, \text{NADPH}} = 0.024,\ K_{i, \text{ATP}} = 0.044,\ K_{\text{G6P}} = 0.027,\ K_{\text{GL6P}} = 0.050,\ K_{\text{NADP}} = 0.019,\ K_{\text{NADPH}} = 0.0105$$ 
# 
# which gives us a value of $K_{\text{G6PDH2r}} = \frac{1000}{\frac{K_{\text{GL6P}}K_{\text{NADPH}}}{K_{\text{G6P}}K_{\text{NADP}}}} = \frac{1000}{\frac{0.050\ *\ 0.0105}{0.027\ *\ 0.019}}$ for the catalyzation step.
# 
# **Total G6PDH2r Concentration:** 0.001 mM

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install --upgrade masspy')


# In[2]:


pip install --upgrade pip


# In[3]:


import mass 
print(mass.__version__)


# ## Import Packages & Set Globals

# In[4]:


import mass 
from mass import (
    MassModel, MassMetabolite, MassReaction, Simulation, Solution,
    plot_simulation, plot_phase_portrait, plot_tiled_phase_portrait, 
    strip_time)
from mass.test import create_test_model
from mass.analysis.linear import nullspace, left_nullspace, matrix_rank
from cobra import DictList
import math
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option('display.max_colwidth', -1)
pd.options.display.float_format = '{:,.3f}'.format

# globals
S_FONT = {"size": "small"}
XL_FONT = {"size": "x-large"}
INF = float("inf")
t0, tf = (0, 1e4)
options={'method': 'LSODA', 'atol': 1e-15}


# ## Part 1: Formation of RBC Model

# ### 1) Merge Pathways and Display overview

# In[5]:


glycolysis = create_test_model("glycolysis")
ppp = create_test_model("pentose_phosphate_pathway")
ampsn = create_test_model("amp_salvage_network")
hemoglobin = create_test_model("hemoglobin")

core_hb = glycolysis.merge(ppp, inplace=False)
core_hb.merge(ampsn, inplace=True, new_model_id="Core-Model")
core_hb.remove_reactions(
    [r for r in core_hb.exchanges 
     if r.id in ["EX_g6p_e", "DM_f6p_e", "DM_g3p_e", "DM_r5p_e",
                 "DM_amp_e", "S_amp_e", "EX_amp_e"]])

core_hb.reactions.PRPPS.subtract_metabolites(
    {core_hb.metabolites.atp_c: -1, core_hb.metabolites.adp_c: 2})
core_hb.reactions.PRPPS.add_metabolites({core_hb.metabolites.amp_c: 1})
core_hb.reactions.PRPPS.kf = 0.619106

core_hb.merge(hemoglobin, inplace=True, new_model_id="Core-RBC-Model")

# Define new order for reactions
new_metabolite_order = ["glc__D_c", "g6p_c", "f6p_c", "fdp_c", "dhap_c","g3p_c", 
                        "_13dpg_c", "_3pg_c", "_2pg_c", "pep_c", "pyr_c", "lac__L_c", 
                        "_6pgl_c", "_6pgc_c", "ru5p__D_c",  "xu5p__D_c", "r5p_c", 
                        "s7p_c", "e4p_c", "ade_c", "adn_c", "imp_c", "ins_c", "hxan_c", 
                        "r1p_c", "prpp_c", "_23dpg_c","hb_c", "hb_1o2_c", "hb_2o2_c", 
                        "hb_3o2_c", "hb_4o2_c", "dhb_c", "nad_c", "nadh_c", "amp_c", 
                        "adp_c", "atp_c", "nadp_c", "nadph_c", "gthrd_c", "gthox_c", 
                        "pi_c", "h_c", "h2o_c", "co2_c", "nh3_c", "o2_c"]
if len(core_hb.metabolites) == len(new_metabolite_order):
    core_hb.metabolites = DictList(core_hb.metabolites.get_by_any(new_metabolite_order))
# Define new order for metabolites
new_reaction_order = ["HEX1", "PGI", "PFK", "FBA", "TPI", "GAPD", "PGK", "PGM", 
                      "ENO", "PYK", "LDH_L", "G6PDH2r", "PGL", "GND", "RPE", 
                      "RPI", "TKT1", "TKT2", "TALA", "ADNK1", "NTD7", "ADA","AMPDA", 
                      "NTD11", "PUNP5", "PPM", "PRPPS", "ADPT", "ADK1", "DPGM", 
                      "DPGase", "HBDPG", "HBO1", "HBO2", "HBO3", "HBO4", "ATPM", 
                      "DM_nadh","GTHOr", "GSHR", "S_glc__D_e", "EX_pyr_e", "EX_lac__L_e",
                      "EX_ade_e", "EX_adn_e", "EX_ins_e", "EX_hxan_e","EX_pi_e", 
                      "EX_h_e", "EX_h2o_e", "EX_co2_e", "EX_nh3_e", "EX_o2_e"]
if len(core_hb.reactions) == len(new_reaction_order):
    core_hb.reactions = DictList(core_hb.reactions.get_by_any(new_reaction_order))
core_hb
ppp.metabolites


# ### 2) Define steady state, PERCs, and equilibrium constants; display on DataFrame

# In[6]:


minspan_paths = np.array([
    [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0],
    [1,-2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 6, 6, 1, 0, 1, 0, 0, 0, 0, 0,13,-3, 3, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,-1, 0, 0, 0, 0, 0, 0, 0,-3, 0, 2, 2, 1, 0, 0,-1, 1, 0, 0, 0, 4, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,-1, 0, 0, 0, 0, 0, 0, 0,-3, 0, 2, 2, 1, 0, 0,-1, 0, 1, 0, 0, 4,-1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1,-1, 0, 0, 0, 0, 0, 0, 0,-3, 0, 2, 2, 1, 0, 0,-1, 0, 1, 0, 0, 4,-1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,-1, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1, 0, 0,-1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
])


independent_fluxes = {core_hb.reactions.S_glc__D_e: 1.12, 
                      core_hb.reactions.DM_nadh: 0.2*1.12, 
                      core_hb.reactions.GSHR : 0.42, 
                      core_hb.reactions.EX_ade_e: -0.014, 
                      core_hb.reactions.ADA: 0.01,
                      core_hb.reactions.EX_adn_e: -0.01, 
                      core_hb.reactions.ADNK1: 0.12, 
                      core_hb.reactions.EX_hxan_e: 0.097,
                      core_hb.reactions.DPGM: 0.441}

ssfluxes = core_hb.compute_steady_state_fluxes(minspan_paths, 
                                               independent_fluxes,
                                               update_reactions=True)

percs = core_hb.calculate_PERCs(update_reactions=True)
core_hb.reactions.EX_o2_e.kf = 509726
core_hb.reactions.HBDPG.kf =519613
core_hb.reactions.HBO1.kf = 506935
core_hb.reactions.HBO2.kf = 511077
core_hb.reactions.HBO3.kf = 509243
core_hb.reactions.HBO4.kf = 501595

sim = Simulation(core_hb)
sim.find_steady_state_model(core_hb, strategy="simulate", 
                            update_initial_conditions=True, 
                            update_reactions=True);


# In[7]:


pd.DataFrame([[r.steady_state_flux for r in core_hb.reactions],
              [r.Keq for r in core_hb.reactions],
              [r.kf for r in core_hb.reactions]],
             index=[r"$\textbf{v}_{\mathrm{stst}}$", r"$K_{eq}$", r"$k_{f}$"],
             columns=[r.id for r in core_hb.reactions])


# ### 3) Verify that Model is in Steady State

# In[8]:


conc_sol, flux_sol = sim.simulate_model(core_hb, time=(t0, tf))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);

plot_simulation(conc_sol, ax=ax, legend="right outside", 
                plot_function="loglog", 
                xlabel="Time (hr)", ylabel="Concentration (mM)", 
                title=("Concentration Profile", XL_FONT));
plt.show()


# ### 4) Add Pre-Built Module

# In[9]:


PFK = create_test_model("PFK")
core_PFK = core_hb.merge(PFK, inplace=False)
core_PFK.remove_reactions(core_PFK.reactions.PFK)
sim = Simulation(core_PFK)
sim.find_steady_state_model(core_PFK, strategy="simulate", 
                            update_initial_conditions=True, 
                            update_reactions=True, **options);


# In[10]:


conc_sol, flux_sol = sim.simulate_model(core_PFK, time=(t0, tf),  **options)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), constrained_layout=True);

plot_simulation(conc_sol, ax=ax, legend="right outside", 
                plot_function="loglog", xlim=(t0, tf), ylim=(1e-10, 1e1),
                xlabel="Time (hr)", ylabel="Concentration (mM)", 
                title=("Concentration Profile", XL_FONT));
plt.show()


# ## Part 2: Enzyme Module Construction

# ### 5) Identify metabolites and reactions of module

# #### Define MassModel Object

# In[11]:


G6PDH = MassModel("G6PDH", matrix_type="DataFrame", dtype=np.int64)


# #### Define Normal Metabolites

# In[12]:


# define metabolites:

nadph_c = MassMetabolite(
    id = "nadph_c",
    name = "Nicotinamide adenine dinucleotide phosphate - reduced",
    formula = "C21H26N7O17P3",
    charge = -4,
    compartment="c")

atp_c = MassMetabolite(
    id = "atp_c",
    name = "ATP",
    formula = "C10H12N5O13P3",
    charge = -4,
    compartment="c")

g6p_c = MassMetabolite(
    id = "g6p_c",
    name = "D-Glucose 6-phosphate",
    formula = "C6H11O9P",
    charge = -2,
    compartment="c")

_6pgl_c = MassMetabolite(
    id = "_6pgl_c",
    name = "6-phospho-D-glucono-1,5-lactone",
    formula = "C6H9O9P",
    charge = -2,
    compartment="c")

nadp_c = MassMetabolite(
    id = "nadp_c",
    name = "Nicotinamide adenine dinucleotide phosphate",
    formula = "C21H25N7O17P3",
    charge = -3,
    compartment="c")

h_c = MassMetabolite(
    id = "h_c",
    name = "H+",
    formula = "H",
    charge = 1,
    compartment="c")

# identify compartment c
G6PDH.compartments = {"c": "Cytosol"}


# #### Define Unbound and Bound Forms of G6PDH

# In[13]:


# define various forms of our enzyme

# relaxed
g6pdh2r_c = MassMetabolite(
    id = "g6pdh2r_c",
    name = "Glucose-6-phosphate dehydrogenase",
    formula = "[G6PDH]",
    charge = 0,
    compartment="c")

# bound to NADPH
g6pdh2r_nadph_c = MassMetabolite(
    id = "g6pdh2r_nadph_c",
    name = "Glucose-6-phosphate dehydrogenase-NADPH complex",
    formula = "[G6PDH]-C21H26N7O17P3",
    charge = -4,
    compartment="c")

# bound to ATP
g6pdh2r_atp_c = MassMetabolite(
    id = "g6pdh2r_atp_c",
    name = "Glucose-6-phosphate dehydrogenase-ATP complex",
    formula = "[G6PDH]-C10H12N5O13P3",
    charge = -4,
    compartment="c")

# bound to G6P
g6pdh2r_g6p_c = MassMetabolite(
    id = "g6pdh2r_g6p_c",
    name = "Glucose-6-phosphate dehydrogenase-G6P complex",
    formula = "[G6PDH]-C6H11O9P",
    charge = -2,
    compartment="c")

# bound to 6PGL 
g6pdh2r_6pgl_c = MassMetabolite(
    id = "g6pdh2r_6pgl_c",
    name ="Glucose-6-phosphate dehydrogenase-6PGL complex",
    formula = "[G6PDH]-C6H9O9P",
    charge = -2,
    compartment="c")

# bound to G6P AND NADP
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(flux_sol, observable = fluxes, plot_function = "semilogx", legend = "right outside",
                xlabel="Time (hr)", ylabel="Flux (mM/hr)", xlim=(t0, tf), 
                title = ("Fluxes of G6PDH2r_3 & G6PDH2r_5 After 20% Increase in ATP Usage in G6PDH Mahidol Model", XL_FONT))
plt.show() = MassMetabolite(
    id = "g6pdh2r_g6p_nadp_c",
    name = "Glucose-6-phosphate dehydrogenase-G6P-NADP complex",
    formula = "[G6PDH]-C6H11O9P-C21H25N7O17P3",
    charge = -5,
    compartment="c")

# bound to 6PGL AND NADPH
g6pdh2r_6pgl_nadph_c = MassMetabolite(
    id = "g6pdh2r_6pgl_nadph_c",
    name = "Glucose-6-phosphate dehydrogenase-6PGL-NADPH complex",
    formula = "[G6PDH]-C6H9O9P-C21H26N7O17P3",
    charge = -6,
    compartment="c")

# create a metabolite list
metabolite_list = [nadph_c, atp_c, g6p_c, _6pgl_c, nadp_c,
                   h_c, g6pdh2r_c, g6pdh2r_nadph_c, g6pdh2r_atp_c,
                   g6pdh2r_g6p_c, g6pdh2r_6pgl_c, g6pdh2r_g6p_nadp_c,
                   g6pdh2r_6pgl_nadph_c]

# add metabolites to model
G6PDH.add_metabolites(metabolite_list)
G6PDH.metabolites


# #### Define Reactions

# In[14]:


# define reactions
# add metabolites to each reaction

# binding of G6PDH to NADPH
G6PDH2r_1 = MassReaction(
    id = "G6PDH2r_1",
    name = "G6PDH-NADPH Binding",
    subsystem = "G6PDH2r")
G6PDH2r_1.add_metabolites({
    g6pdh2r_c : -1,
    nadph_c : -1,
    g6pdh2r_nadph_c : 1})

# binding of G6PDH to ATP
G6PDH2r_2 = MassReaction(
    id = "G6PDH2r_2",
    name = "G6PDH-ATP Binding",
    subsystem = "G6PDH2r")
G6PDH2r_2.add_metabolites({
    g6pdh2r_c : -1,
    atp_c : -1,
    g6pdh2r_atp_c : 1})

# binding of G6PDH to G6P
G6PDH2r_3 = MassReaction(
    id = "G6PDH2r_3",
    name = "G6PDH-G6P binding",
    subsystem = "G6PDH2r")
G6PDH2r_3.add_metabolites({
    g6pdh2r_c : -1,
    g6p_c : -1,
    g6pdh2r_g6p_c : 1})

# binding of G6PDH to 6PGL
G6PDH2r_4 = MassReaction(
    id = "G6PDH2r_4",
    name = "G6PDH-6PGL binding",
    subsystem = "G6PDH2r")
G6PDH2r_4.add_metabolites({
    g6pdh2r_6pgl_c : -1,
    _6pgl_c : 1,
    g6pdh2r_c : 1})

# binding of G6PDH-G6P complex to NADP
G6PDH2r_5 = MassReaction(
    id = "G6PDH2r_5",
    name = "G6PDH-G6P-NADP binding",
    subsystem = "G6PDH2r")
G6PDH2r_5.add_metabolites({
    g6pdh2r_g6p_c : -1,
    nadp_c : -1,
    g6pdh2r_g6p_nadp_c : 1})

# binding of G6PDH-6PGL complex to NADPH
G6PDH2r_6 = MassReaction(
    id = "G6PDH2r_6",
    name = "G6PDH-6PGL-NADPH binding",
    subsystem = "G6PDH2r")
G6PDH2r_6.add_metabolites({
    g6pdh2r_6pgl_nadph_c : -1,
    g6pdh2r_6pgl_c : 1, 
    nadph_c : 1})

# reduction of G6PDH-G6P-NADP complex 
G6PDH2r_7 = MassReaction(
    id = "G6PDH2r_7",
    name = "G6PDH-G6P-NADP reduction",
    subsystem = "G6PDH2r")
G6PDH2r_7.add_metabolites({
    g6pdh2r_g6p_nadp_c : -1,
    g6pdh2r_6pgl_nadph_c : 1,
    h_c : 1})

# make reaction list
reaction_list = [G6PDH2r_1, G6PDH2r_2, G6PDH2r_3, G6PDH2r_4,
                G6PDH2r_5, G6PDH2r_6, G6PDH2r_7]

# add reaction to model
G6PDH.add_reactions(reaction_list)


# ### 6) Determine Enzyme Reaction Rate 

# In[15]:


enzyme_forms = [sym.Symbol(met.id) for met in G6PDH.metabolites if "g6pdh" in met.id]
conc_subs = {sym.Symbol(met.id): 1 for met in G6PDH.metabolites if "g6pdh" not in met.id}
ode_dict = {sym.Symbol(met.id): sym.Eq(ode)
            for met, ode in strip_time(G6PDH.odes).items()
            if "g6pdh" in met.id}


# In[16]:


enzyme_forms.reverse()

enzyme_solutions = {}
for enzyme_form in enzyme_forms:
    if "g6pdh2r_c" == str(enzyme_form):
        continue
    sol = sym.solveset(ode_dict[enzyme_form].subs(enzyme_solutions), enzyme_form)
    enzyme_solutions[enzyme_form] = list(sol).pop()
    enzyme_solutions.update({enzyme_form: sol.subs(enzyme_solutions) 
                             for enzyme_form, sol in enzyme_solutions.items()})


# In[17]:


abbrev_dict = {"G6PDH2:"}
catalyzation_reactions = ["G6PDH2r_7"]
rate_sym = sym.Symbol("v_G6PDH2r")
rate_equation = rate_sym - sym.simplify(sum(strip_time([
    G6PDH.reactions.get_by_id(rxn).rate for rxn in catalyzation_reactions])))
sym.pprint(rate_equation)


# ### 7) Symbolically Determine Enzyme Concentrations Correctly 

# In[18]:


sol = sym.solveset(rate_equation.subs(enzyme_solutions), "g6pdh2r_c")
enzyme_solutions[sym.Symbol("g6pdh2r_c")] = list(sol).pop()
enzyme_solutions = {met: sym.simplify(solution.subs(enzyme_solutions))
                    for met, solution in enzyme_solutions.items()}


# ### 8) Define Equilibrium Constants and Substitute into Equations

# In[19]:


# binding = inverse
# dissociation = same number
glycolysis_ppp = glycolysis.merge(ppp, inplace=False, new_model_id = "Glycolysis & ppp")
glycolysis_ppp.remove_reactions([r for r in glycolysis_ppp.exchanges
                                  if r.id in ["EX_g6p_e", "DM_f6p_e", "DM_g3p_e", "DM_r5p_e"]])

numerical_values = {rate_sym: glycolysis_ppp.reactions.G6PDH2r.ssflux}
numerical_values.update({
    sym.Symbol(met.id): glycolysis_ppp.initial_conditions[glycolysis_ppp.metabolites.get_by_id(met.id)]
    for met in glycolysis_ppp.metabolites if met.id in G6PDH.metabolites})

reaction_list = ["G6PDH2r_1","G6PDH2r_2","G6PDH2r_3","G6PDH2r_4","G6PDH2r_5", "G6PDH2r_6","G6PDH2r_7"]

Keq_values = [round(v, 6) for v in [1/0.024, 1/0.044, 1/0.027, 0.050, 1/0.019, 0.0105, (1000)/((0.050*0.0105)/(0.027*0.019))]]

Keq_values = dict(zip(["Keq_" + p for p in reaction_list], Keq_values))
numerical_values.update(Keq_values)

enzyme_solutions = {met: sym.simplify(solution.subs(numerical_values))
                    for met, solution in enzyme_solutions.items()}
numerical_values


# In[20]:


rate_equation = sym.simplify(rate_equation.subs(numerical_values))
sym.pprint(sym.N(rate_equation, 3))


# ### 9) Display G6PDH in terms of enzymes and rate constant

# In[21]:


total_sym = sym.Symbol("G6PDH2r-Total")
total_equation = sym.Eq(total_sym, sum(enzyme_forms))
sym.pprint(total_equation)


# In[22]:


total_equation = sym.simplify(total_equation.subs(enzyme_solutions))
sym.pprint(sym.N(total_equation, 3))


# ### 10) Determine Rate Constant Correctly with Optimization Error less Than 1e-6

# In[23]:


args = ("kf_G6PDH2r_3", "kf_G6PDH2r_4", "kf_G6PDH2r_5","kf_G6PDH2r_6","kf_G6PDH2r_7")
total_conc = {total_sym: .001}

total_constraint = sym.Abs((total_equation.lhs - total_equation.rhs).subs(total_conc))
sym.pprint(total_constraint)

obj_fun = lambda x: sym.lambdify(args, total_constraint)(*x)


# In[24]:


t_forms = [sym.Symbol("g6pdh2r_nadph_c"), sym.Symbol("g6pdh2r_atp_c")]
r_T_cons = sym.simplify(sym.simplify(sum(t_forms)/sum(enzyme_forms)).subs(enzyme_solutions))


# In[25]:


bounds = ((1, 1e9), (1, 1e9), (1, 1e9),(1, 1e9),(1, 1e9))


sol = optimize.minimize(
    obj_fun, [5e-6, 5e-6, 2e-6, 5e-6, 6e-6], method="trust-constr", bounds=bounds, 
    options={"gtol": 1e-12, "xtol": 1e-12, "maxiter": 1e3, "disp": True})

parameter_solutions = dict(zip(args, [round(x) for x in sol.x]))

print(parameter_solutions)
print(sol.success)


# ### 11) Display Optimization Error and T-Fraction

# In[26]:


error = total_constraint.subs(parameter_solutions)
print("Optimization Error: {0:.12f}".format(error))
print("Enzyme Fraction in T state: {0:.3f}".format(r_T_cons.subs(parameter_solutions)))


# ### 12) Set Module Initial Conditions and Display

# In[27]:


for met in G6PDH.metabolites:
    if "g6pdh2r" in met.id:
        sol = float(enzyme_solutions[sym.Symbol(str(met))].subs(parameter_solutions))
        met.ic = round(sol, 12)
    else:
        glyc_ppp_met = glycolysis_ppp.metabolites.get_by_id(met.id)
        met.ic = glycolysis_ppp.initial_conditions[glyc_ppp_met]

G6PDH.set_initial_conditions()

G6PDH2r_1.Keq = 41.666667
G6PDH2r_1.kf = 1e6

G6PDH2r_2.Keq = 22.727273
G6PDH2r_2.kf = 1e6

G6PDH2r_3.Keq = 37.037037
G6PDH2r_3.kf = 2785811.0

G6PDH2r_4.Keq = 0.05
G6PDH2r_4.kf = 233366.0

G6PDH2r_5.Keq = 52.631579
G6PDH2r_5.kf = 29150780.0

G6PDH2r_6.Keq = 0.0105
G6PDH2r_6.kf = 2460764.0

G6PDH2r_7.Keq = 977.142857
G6PDH2r_7.kf = 7511864.0

G6PDH.initial_conditions
G6PDH.set_initial_conditions()
for param_type, param_dict in G6PDH.parameters.items():
    print("%s: %s" %(param_type, param_dict))


# ### 13) QC/QA

# In[28]:


mass.qcqa_model(G6PDH, parameters=True, concentrations=True, 
                fluxes=False, superfluous=True, elemental=True)


# ### 14) Add Module to Core RBC and Simulate to Steady State

# In[29]:


# merge with core RBC
core_hb_G6PDH = core_hb.merge(G6PDH, new_model_id="FINAL RBC")
core_hb_G6PDH.remove_reactions(core_hb_G6PDH.reactions.get_by_id("G6PDH2r"))
core_hb_G6PDH.remove_fixed_concentrations([met for met in core_hb_G6PDH.fixed_concentrations if met not in core_hb_G6PDH.external_metabolites])
# Setup simulation object, ensure model is at steady state
sim = Simulation(core_hb_G6PDH)
sim.find_steady_state_model(core_hb_G6PDH, strategy="simulate",
                            update_initial_conditions=True, 
                            update_reactions=True, **options)

conc_sol, flux_sol = sim.simulate_model(core_hb_G6PDH, time=(t0, tf), **options)
conc_sol.preview_time_profile

plot_simulation(conc_sol, ax=ax, legend="right outside", 
                plot_function="loglog", xlim=(t0, tf), ylim=(1e-10, 1e2),
                xlabel="Time (hr)", ylabel="Concentration (mM)", 
                title=("Concentration Profile of Healthy RBC", XL_FONT))
plt.show()


# In[30]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_phase_portrait(conc_sol, x="nadp_c", y="g6pdh2r_c", ax = ax,
                    xlabel="NADP+ (mM)", ylabel="G6PDH (mM)", time_poi=[t0, 1e0, 1e1, tf],
                   title = ("Phase Diagram of G6PDH vs NADP+ in Healthy RBC", XL_FONT))
plt.show()


# In[31]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_phase_portrait(conc_sol, x="g6p_c", y="g6pdh2r_c", ax = ax,
                    xlabel="G6P (mM)", ylabel="G6PDH (mM)", time_poi=[t0, 1e0, 1e1, tf],
                   title = ("Phase Diagram of G6PDH vs G6P in Healthy RBC", XL_FONT))
plt.show()


# In[32]:


fluxes = ["G6PDH2r_3", "G6PDH2r_5"]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(flux_sol, observable = fluxes, plot_function = "semilogx", legend = "right outside",
                xlabel="Time (hr)", ylabel="Flux (mM/hr)", ax = ax,
               title = ("Fluxes of G6PDH2r_3 & G6PDH2r_5 in Healty RBC", XL_FONT))
plt.show()


# In[33]:


conc_sol, flux_sol = sim.simulate_model(core_hb_G6PDH, time = (t0,tf),
                                                 perturbations={"nadph_c.ic":"[nadph_c.ic]*1.25"},
                                                 interpolate=False, **options)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(conc_sol, legend="right outside", ax = ax,
                plot_function="semilogx", xlim=(t0, tf), ylim=(1e-10, 10),
                xlabel="Time (hr)", ylabel="Concentration (mM)", 
                title=("Concentration Profile After 20% Increase in ATP Usage in Healthy RBC", XL_FONT))
plt.show()


# In[34]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_phase_portrait(conc_sol, x="nadp_c", y="g6pdh2r_c", ax = ax,
                    xlabel="NADP+ (mM)", ylabel="G6PDH (mM)", time_poi=[t0, 1e0, 1e1, tf],
                   title = ("Phase Diagram of G6PDH vs NADP+ After 20% Increase in ATP Usage in Healthy RBC", XL_FONT))
plt.show()


# In[35]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_phase_portrait(conc_sol, x="g6p_c", y="g6pdh2r_c", ax = ax, 
                    xlabel="G6P (mM)", ylabel="G6PDH (mM)", time_poi=[t0, 1e0, 1e1, tf],
                   title = ("Phase Diagram of G6PDH vs G6P After 20% Increase in ATP Usage in Healthy RBC", XL_FONT))
plt.show()


# In[36]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(flux_sol, observable = fluxes, plot_function = "semilogx", legend = "right outside",
                xlabel="Time (hr)", ylabel="Flux (mM/hr)", xlim=(t0, tf), ylim=(0, .3),
               title = ("Fluxes of G6PDH2r_3 & G6PDH2r_5 After 20% Increase in ATP Usage in Healthy RBC", XL_FONT))
plt.show()


# In[37]:


conc_sol, flux_sol = sim.simulate_model(core_hb_G6PDH, time = (t0,tf),
                                                 perturbations={"nadph_c.ic":"[nadph_c.ic]*1.25"},
                                                 interpolate=False, **options)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(conc_sol, ax=ax, legend="right outside",
                plot_function="semilogx", xlim=(t0, tf), ylim=(1e-10, 10),
                xlabel="Time (hr)", ylabel="Concentration (mM)", 
                title=("Concentration Profile", XL_FONT))
plt.show()


# In[38]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(conc_sol, observable = ["nadph_c"], plot_function = "semilogx", legend = "right outside",
                xlabel="Time (hr)", ylabel="Concentration (mM)", ax = ax,
                title = ("Concentration Change of NADPH in Mahidol After Perturbation", XL_FONT))
plt.show()


# ## Part 3: Enzyme Module Modification

# ### 15) Introduce Modification #1

# In[39]:


G6PDH_modified = MassModel("G6PDH", matrix_type="DataFrame", dtype=np.int64)


# In[40]:


# define metabolites:

nadph_c = MassMetabolite(
    id = "nadph_c",
    name = "Nicotinamide adenine dinucleotide phosphate - reduced",
    formula = "C21H26N7O17P3",
    charge = -4,
    compartment="c")

atp_c = MassMetabolite(
    id = "atp_c",
    name = "ATP",
    formula = "C10H12N5O13P3",
    charge = -4,
    compartment="c")

g6p_c = MassMetabolite(
    id = "g6p_c",
    name = "D-Glucose 6-phosphate",
    formula = "C6H11O9P",
    charge = -2,
    compartment="c")

_6pgl_c = MassMetabolite(
    id = "_6pgl_c",
    name = "6-phospho-D-glucono-1,5-lactone",
    formula = "C6H9O9P",
    charge = -2,
    compartment="c")

nadp_c = MassMetabolite(
    id = "nadp_c",
    name = "Nicotinamide adenine dinucleotide phosphate",
    formula = "C21H25N7O17P3",
    charge = -3,
    compartment="c")

h_c = MassMetabolite(
    id = "h_c",
    name = "H+",
    formula = "H",
    charge = 1,
    compartment="c")

# identify compartment c
G6PDH_modified.compartments = {"c": "Cytosol"}


# In[41]:


# define various forms of our enzyme

# relaxed
g6pdh2r_c = MassMetabolite(
    id = "g6pdh2r_c",
    name = "Glucose-6-phosphate dehydrogenase",
    formula = "[G6PDH]",
    charge = 0,
    compartment="c")

# bound to NADPH
g6pdh2r_nadph_c = MassMetabolite(
    id = "g6pdh2r_nadph_c",
    name = "Glucose-6-phosphate dehydrogenase-NADPH complex",
    formula = "[G6PDH]-C21H26N7O17P3",
    charge = -4,
    compartment="c")

# bound to ATP
g6pdh2r_atp_c = MassMetabolite(
    id = "g6pdh2r_atp_c",
    name = "Glucose-6-phosphate dehydrogenase-ATP complex",
    formula = "[G6PDH]-C10H12N5O13P3",
    charge = -4,
    compartment="c")

# bound to G6P
g6pdh2r_g6p_c = MassMetabolite(
    id = "g6pdh2r_g6p_c",
    name = "Glucose-6-phosphate dehydrogenase-G6P complex",
    formula = "[G6PDH]-C6H11O9P",
    charge = -2,
    compartment="c")

# bound to 6PGL 
g6pdh2r_6pgl_c = MassMetabolite(
    id = "g6pdh2r_6pgl_c",
    name ="Glucose-6-phosphate dehydrogenase-6PGL complex",
    formula = "[G6PDH]-C6H9O9P",
    charge = -2,
    compartment="c")

# bound to G6P AND NADP
g6pdh2r_g6p_nadp_c = MassMetabolite(
    id = "g6pdh2r_g6p_nadp_c",
    name = "Glucose-6-phosphate dehydrogenase-G6P-NADP complex",
    formula = "[G6PDH]-C6H11O9P-C21H25N7O17P3",
    charge = -5,
    compartment="c")

# bound to 6PGL AND NADPH
g6pdh2r_6pgl_nadph_c = MassMetabolite(
    id = "g6pdh2r_6pgl_nadph_c",
    name = "Glucose-6-phosphate dehydrogenase-6PGL-NADPH complex",
    formula = "[G6PDH]-C6H9O9P-C21H26N7O17P3",
    charge = -6,
    compartment="c")

# create a metabolite list
metabolite_list = [nadph_c, atp_c, g6p_c, _6pgl_c, nadp_c,
                   h_c, g6pdh2r_c, g6pdh2r_nadph_c, g6pdh2r_atp_c,
                   g6pdh2r_g6p_c, g6pdh2r_6pgl_c, g6pdh2r_g6p_nadp_c,
                   g6pdh2r_6pgl_nadph_c]

# add metabolites to model
G6PDH_modified.add_metabolites(metabolite_list)
G6PDH_modified.metabolites


# In[42]:


# define reactions
# add metabolites to each reaction

# binding of G6PDH to NADPH
G6PDH2r_1 = MassReaction(
    id = "G6PDH2r_1",
    name = "G6PDH-NADPH Binding",
    subsystem = "G6PDH2r")
G6PDH2r_1.add_metabolites({
    g6pdh2r_c : -1,
    nadph_c : -1,
    g6pdh2r_nadph_c : 1})

# binding of G6PDH to ATP
G6PDH2r_2 = MassReaction(
    id = "G6PDH2r_2",
    name = "G6PDH-ATP Binding",
    subsystem = "G6PDH2r")
G6PDH2r_2.add_metabolites({
    g6pdh2r_c : -1,
    atp_c : -1,
    g6pdh2r_atp_c : 1})

# binding of G6PDH to G6P
G6PDH2r_3 = MassReaction(
    id = "G6PDH2r_3",
    name = "G6PDH-G6P binding",
    subsystem = "G6PDH2r")
G6PDH2r_3.add_metabolites({
    g6pdh2r_c : -1,
    g6p_c : -1,
    g6pdh2r_g6p_c : 1})

# binding of G6PDH to 6PGL
G6PDH2r_4 = MassReaction(
    id = "G6PDH2r_4",
    name = "G6PDH-6PGL binding",
    subsystem = "G6PDH2r")
G6PDH2r_4.add_metabolites({
    g6pdh2r_6pgl_c : -1,
    _6pgl_c : 1,
    g6pdh2r_c : 1})

# binding of G6PDH-G6P complex to NADP
G6PDH2r_5 = MassReaction(
    id = "G6PDH2r_5",
    name = "G6PDH-G6P-NADP binding",
    subsystem = "G6PDH2r")
G6PDH2r_5.add_metabolites({
    g6pdh2r_g6p_c : -1,
    nadp_c : -1,
    g6pdh2r_g6p_nadp_c : 1})

# binding of G6PDH-6PGL complex to NADPH
G6PDH2r_6 = MassReaction(
    id = "G6PDH2r_6",
    name = "G6PDH-6PGL-NADPH binding",
    subsystem = "G6PDH2r")
G6PDH2r_6.add_metabolites({
    g6pdh2r_6pgl_nadph_c : -1,
    g6pdh2r_6pgl_c : 1, 
    nadph_c : 1})

# reduction of G6PDH-G6P-NADP complex 
G6PDH2r_7 = MassReaction(
    id = "G6PDH2r_7",
    name = "G6PDH-G6P-NADP reduction",
    subsystem = "G6PDH2r")
G6PDH2r_7.add_metabolites({
    g6pdh2r_g6p_nadp_c : -1,
    g6pdh2r_6pgl_nadph_c : 1,
    h_c : 1})

# make reaction list
reaction_list = [G6PDH2r_1, G6PDH2r_2, G6PDH2r_3, G6PDH2r_4,
                G6PDH2r_5, G6PDH2r_6, G6PDH2r_7]

# add reaction to model
G6PDH_modified.add_reactions(reaction_list)


# In[43]:


# In 50
enzyme_forms = [sym.Symbol(met.id) for met in G6PDH_modified.metabolites if "g6pdh" in met.id]
conc_subs = {sym.Symbol(met.id): 1 for met in G6PDH_modified.metabolites if "g6pdh" not in met.id}
ode_dict = {sym.Symbol(met.id): sym.Eq(ode)
            for met, ode in strip_time(G6PDH_modified.odes).items()
            if "g6pdh" in met.id}


# In[44]:


# In 51
enzyme_forms.reverse()

enzyme_solutions = {}
for enzyme_form in enzyme_forms:
    if "g6pdh2r_c" == str(enzyme_form):
        continue
    sol = sym.solveset(ode_dict[enzyme_form].subs(enzyme_solutions), enzyme_form)
    enzyme_solutions[enzyme_form] = list(sol).pop()
    enzyme_solutions.update({enzyme_form: sol.subs(enzyme_solutions) 
                             for enzyme_form, sol in enzyme_solutions.items()})


# In[45]:


# In 52
catalyzation_reactions = ["G6PDH2r_7"]
rate_sym = sym.Symbol("v_G6PDH2r")
rate_equation = rate_sym - sym.simplify(sum(strip_time([
    G6PDH_modified.reactions.get_by_id(rxn).rate for rxn in catalyzation_reactions])))
sym.pprint(rate_equation)


# In[46]:


# In 53
sol = sym.solveset(rate_equation.subs(enzyme_solutions), "g6pdh2r_c")
enzyme_solutions[sym.Symbol("g6pdh2r_c")] = list(sol).pop()
enzyme_solutions = {met: sym.simplify(solution.subs(enzyme_solutions))
                    for met, solution in enzyme_solutions.items()}


# In[47]:


glyc_ppp = glycolysis.merge(ppp, inplace=False, new_model_id = "Glycolysis & ppp")
glyc_ppp.remove_reactions([r for r in glyc_ppp.exchanges
                                  if r.id in ["EX_g6p_e", "DM_f6p_e", "DM_g3p_e", "DM_r5p_e"]])

numerical_values = {rate_sym: glyc_ppp.reactions.G6PDH2r.ssflux}
numerical_values.update({
    sym.Symbol(met.id): glyc_ppp.initial_conditions[glyc_ppp.metabolites.get_by_id(met.id)]
    for met in glyc_ppp.metabolites if met.id in G6PDH_modified.metabolites})

reaction_list = ["G6PDH2r_1","G6PDH2r_2","G6PDH2r_3","G6PDH2r_4","G6PDH2r_5", "G6PDH2r_6","G6PDH2r_7"]

Keq_values = [round(v, 6) for v in [1/0.024, 1/0.044, 1/0.0507, 0.050, 1/0.00646, 0.0105, (1000)/((0.050*0.0105)/(0.027*0.019))]]

Keq_values = dict(zip(["Keq_" + p for p in reaction_list], Keq_values))
numerical_values.update(Keq_values)

enzyme_solutions = {met: sym.simplify(solution.subs(numerical_values))
                    for met, solution in enzyme_solutions.items()}
numerical_values


# In[48]:


args = ("kf_G6PDH2r_3", "kf_G6PDH2r_4", "kf_G6PDH2r_5","kf_G6PDH2r_6","kf_G6PDH2r_7")
total_conc = {total_sym: .001}

total_constraint = sym.Abs((total_equation.lhs - total_equation.rhs).subs(total_conc))
sym.pprint(total_constraint)

obj_fun = lambda x: sym.lambdify(args, total_constraint)(*x)


# In[49]:


t_forms = [sym.Symbol("g6pdh2r_nadph_c"), sym.Symbol("g6pdh2r_atp_c")]
r_T_cons = sym.simplify(sym.simplify(sum(t_forms)/sum(enzyme_forms)).subs(enzyme_solutions))


# In[50]:


bounds = ((1, 1e9), (1, 1e9), (1, 1e9),(1, 1e9),(1, 1e9))

sol = optimize.minimize(
    obj_fun, [5e-6, 5e-6, 2e-6, 5e-6, 6e-6], method="trust-constr", bounds=bounds,
    options={"gtol": 1e-12, "xtol": 1e-12, "maxiter": 1e3, "disp": True})

parameter_solutions = dict(zip(args, [round(x) for x in sol.x]))

print(parameter_solutions)
print(sol.success)


# In[51]:


error = total_constraint.subs(parameter_solutions)
print("Optimization Error: {0:.12f}".format(error))
print("Enzyme Fraction in T state: {0:.3f}".format(r_T_cons.subs(parameter_solutions)))


# In[52]:


for met in G6PDH_modified.metabolites:
    if "g6pdh2r" in met.id:
        sol = float(enzyme_solutions[sym.Symbol(str(met))].subs(parameter_solutions))
        met.ic = round(sol, 12)
    else:
        glyc_ppp_met = glyc_ppp.metabolites.get_by_id(met.id)
        met.ic = glyc_ppp.initial_conditions[glyc_ppp_met]

G6PDH_modified.set_initial_conditions()

G6PDH2r_1.Keq = 41.666667
G6PDH2r_1.kf = 1e6

G6PDH2r_2.Keq = 22.727273
G6PDH2r_2.kf = 1e6

G6PDH2r_3.Keq = 19.723866
G6PDH2r_3.kf = 2785811.0

G6PDH2r_4.Keq = 0.05
G6PDH2r_4.kf = 233366.0

G6PDH2r_5.Keq = 154.798762
G6PDH2r_5.kf = 29150780.0

G6PDH2r_6.Keq = 0.0105
G6PDH2r_6.kf = 2460764.0

G6PDH2r_7.Keq = 977.142857
G6PDH2r_7.kf = 7511864.0

G6PDH_modified.set_initial_conditions()
for param_type, param_dict in G6PDH_modified.parameters.items():
    print("%s: %s" %(param_type, param_dict))


# In[53]:


mass.qcqa_model(G6PDH_modified, parameters=True, concentrations=True, 
                fluxes=False, superfluous=True, elemental=True)


# In[54]:


# merge with core RBC
modified_core_hb_G6PDH_2 = core_hb.merge(G6PDH_modified, new_model_id="MODIFIED RBC")
modified_core_hb_G6PDH_2.remove_reactions(modified_core_hb_G6PDH_2.reactions.get_by_id("G6PDH2r"))
modified_core_hb_G6PDH_2.remove_fixed_concentrations([met for met in modified_core_hb_G6PDH_2.fixed_concentrations if met not in modified_core_hb_G6PDH_2.external_metabolites])
# Setup simulation object, ensure model is at steady state
sim = Simulation(modified_core_hb_G6PDH_2)
sim.find_steady_state_model(modified_core_hb_G6PDH_2, strategy="simulate",
                            update_initial_conditions=True, 
                            update_reactions=True, **options)

conc_sol, flux_sol = sim.simulate_model(modified_core_hb_G6PDH_2, time=(t0, tf), **options)
conc_sol.preview_time_profile

plot_simulation(conc_sol, ax=ax, legend="right outside",
                plot_function="loglog", xlim=(t0, tf), ylim=(1e-10, 1e2),
                xlabel="Time (hr)", ylabel="Concentration (mM)", 
                title=("Concentration Profile of RBC with G6PDH Mahidol", XL_FONT))
plt.show()


# In[55]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_phase_portrait(conc_sol, x="nadp_c", y="g6pdh2r_c", ax=ax, 
                    xlabel="NADP+ (mM)", ylabel="G6PDH (mM)", time_poi=[t0, 1e0, 1e1, tf],
                   title = ("Phase Diagram of G6PDH vs NADP+ in RBC with G6PDH Mahidol", XL_FONT))
plt.show()


# In[56]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_phase_portrait(conc_sol, x="g6p_c", y="g6pdh2r_c", ax = ax,
                    xlabel="G6P (mM)", ylabel="G6PDH (mM)", time_poi=[t0, 1e0, 1e1, tf],
                   title = ("Phase Diagram of G6PDH vs G6P in RBC with G6PDH Mahidol", XL_FONT))
plt.show()


# In[57]:


fluxes = ["G6PDH2r_3", "G6PDH2r_5"]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(flux_sol, observable = fluxes, plot_function = "semilogx", legend = "right outside",
                xlabel="Time (hr)", ylabel="Flux (mM/hr)", ax = ax,
               title = ("Fluxes of G6PDH2r_3 & G6PDH2r_5 in RBC with G6PDH Mahidol", XL_FONT))
plt.show()


# In[58]:


conc_sol, flux_sol = sim.simulate_model(modified_core_hb_G6PDH_2, time = (t0,tf),
                                                 perturbations={"nadph_c.ic":"[nadph_c.ic]*1.25"},
                                                 interpolate=False, **options)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(conc_sol, ax=ax, legend="right outside",
                plot_function="semilogx", xlim=(t0, tf), ylim=(1e-10, 10),
                xlabel="Time (hr)", ylabel="Concentration (mM)", 
                title=("Concentration Profile", XL_FONT))
plt.show()


# In[59]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_phase_portrait(conc_sol, x="nadp_c", y="g6pdh2r_c", ax = ax,
                    xlabel="NADP+ (mM)", ylabel="G6PDH (mM)", time_poi=[t0, 1e0, 1e1, tf],
                   title = ("Phase Diagram of G6PDH vs NADP+ After 20% Increase in ATP Usage in G6PDH Mahidol Model", XL_FONT))
plt.show()


# In[60]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_phase_portrait(conc_sol, x="g6p_c", y="g6pdh2r_c", ax = ax,
                    xlabel="G6P (mM)", ylabel="G6PDH (mM)", time_poi=[t0, 1e0, 1e1, tf],
                   title = ("Phase Diagram of G6PDH vs G6P After 20% Increase in ATP Usage in G6PDH Mahidol Model", XL_FONT))
plt.show()


# In[61]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(flux_sol, observable = fluxes, plot_function = "semilogx", legend = "right outside",
                xlabel="Time (hr)", ylabel="Flux (mM/hr)", xlim=(t0, tf), ylim=(0, .3),
                title = ("Fluxes of G6PDH2r_3 & G6PDH2r_5 After 20% Increase in ATP Usage in G6PDH Mahidol Model", XL_FONT))
plt.show()


# In[62]:


conc_sol, flux_sol = sim.simulate_model(modified_core_hb_G6PDH_2, time = (t0,tf),
                                                 perturbations={"nadph_c.ic":"[nadph_c.ic]*1.25"},
                                                 interpolate=False, **options)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(conc_sol, ax=ax, legend="right outside",
                plot_function="semilogx", xlim=(t0, tf), ylim=(1e-10, 10),
                xlabel="Time (hr)", ylabel="Concentration (mM)", 
                title=("Concentration Profile", XL_FONT))
plt.show()


# In[63]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4), constrained_layout=True);
plot_simulation(conc_sol, observable = ["nadph_c"], plot_function = "semilogx", legend = "right outside",
                xlabel="Time (hr)", ylabel="Concentration (mM)", ax = ax,
                title = ("Concentration Change of NADPH in Mahidol After Perturbation", 
                         XL_FONT))
plt.show()


# ## Export Models for Analysis
# To assist in the export/import process, some code and instructions have been provided for you. Please keep in mind that the code assumes that the model objects will be exported/imported from the same directory where this notebook exits. Uncomment the code to use it.
# 
# ### Set Models
# * Set the variable ``MY_BASE_MODEL`` to your current base model object in order to export it to the current directory.
# * Set the variable ``MY_MODIFIED_MODEL`` to your current base model object in order to export it to the current directory.

# In[64]:


MY_BASE_MODEL = core_hb_G6PDH
MY_MODIFIED_MODEL = modified_core_hb_G6PDH_2


# ### Export Models

# In[65]:


from mass.io import json
for model, model_type in zip([MY_BASE_MODEL, MY_MODIFIED_MODEL], ["BASE", "MODIFIED"]):
     filepath = "./{0}_{1}.json".format(model.id, model_type)
     json.save_json_model(model, filepath)
     print("Saved {0} Model with ID: {1}".format(model_type, model.id))
model.remove_fixed_concentrations([met for met in model.fixed_concentrations if met not in model.external_metabolites])


# In[ ]:




