#####QUERIES FOR BOTH EVALUATION EXPERIMENTS - AI, and SciFact viral mechanism/effect statements
query_ai_uses = [ 

    {"Description":"Artificial Intelligence models for COVID-19 and related areas", "x":["artificial intelligence"],"y":[],"bidirect":False},

    {"Description":"Machine learning models for COVID-19 and related areas", "x":["machine learning"],"y":[],"bidirect":False},

    {"Description":"Statistical models for COVID-19 and related areas", "x":["statistical models"],"y":[],"bidirect":False},

    {"Description":"Predictive models for COVID-19 and related areas", "x":["predictive models"],"y":[],"bidirect":False},
    
    {"Description":"Graph Neural Network models for COVID-19 and related areas", "x":["Graph Neural Network model"],"y":[],"bidirect":False},

    {"Description":"Convolutional Neural Network models for COVID-19 and related areas", "x":["Convolutional Neural Network model"],"y":[],"bidirect":False},

    {"Description":"Recurrent Neural Network models for COVID-19 and related areas", "x":["Recurrent Neural Network model"],"y":[],"bidirect":False},

    {"Description":"Reinforcement learning for COVID-19 and related areas", "x":["reinforcement learning"],"y":[],"bidirect":False},

    {"Description":"Image analysis for COVID-19 and related areas", "x":["image analysis"],"y":[],"bidirect":False},

    {"Description":"Text analysis for COVID-19 and related areas", "x":["text analysis"],"y":[],"bidirect":False},

    {"Description":"Speech analysis for COVID-19 and related areas", "x":["speech analysis"],"y":[],"bidirect":False}]

queries_scifact = [

    {"Description":"Remdesevir has exhibited favorable clinical responses when used as a treatment for coronavirus.", "x":["Remdesevir"],"y":["SARS-CoV-2","coronavirus","COVID-19"],"bidirect":False},

    {"Description":"Lopinavir / ritonavir have exhibited favorable clinical responses when used as a treatment for coronavirus.", "x":["Lopinavir","Ritonavir"],"y":["SARS-CoV-2","coronavirus","COVID-19"],"bidirect":False},

    {"Description":"Aerosolized SARS-CoV-2 viral particles can travel further than 6 feet.","x":["Air","Aerosols","Droplets","Particles","Distance"],"y":["SARS-CoV-2 transmission"],"bidirect":False},

    {"Description":"Chloroquine has shown antiviral efficacy against SARS-CoV-2 in vitro through interference with the ACE2-receptor mediated endocytosis.","x":["Chloroquine"],"y":["ACE2-receptor","Endocytosis","interference with the ACE2-receptor mediated endocytosis."],"bidirect":False},

    {"Description":"Lymphopenia is associated with severe COVID-19 disease.","x":["Lymphopenia"],"y":["severe COVID-19 disease","COVID-19"],"bidirect":True},

    {"Description":"Bilateral ground glass opacities are often seen on chest imaging in COVID-19 patients.","x":["Bilateral ground glass opacities"],"y":["chest imaging in COVID-19 patients"],"bidirect":True},

    {"Description":"Cardiac injury is common in critical cases of COVID-19.","x":["COVID-19"],"y":["Cardiac injury"],"bidirect":False},

    {"Description":"Cats are carriers of SARS-CoV-2.","x":["Cats"],"y":["SARS-CoV-2"],"bidirect":True},

    {"Description":"Diabetes is a common comorbidity seen in COVID-19 patients.","x":["Diabetes"],"y":["COVID-19"],"bidirect":True},

    {"Description":"The coronavirus cannot thrive in warmer climates.","x":["warmer climates"],"y":["coronavirus"],"bidirect":True},

    {"Description":"SARS-CoV-2 binds ACE2 receptor to gain entry into cells.","x":["SARS-CoV-2"],"y":["binds ACE2 receptor","binds ACE2 receptor to gain entry into cells"],"bidirect":True}
    ]