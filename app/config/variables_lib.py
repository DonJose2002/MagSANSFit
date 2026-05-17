from app.prototypes.variable import Variable
"""
Stores all possible fit variables names. In order: Type name, display name, internal name
"""
ALL_VARIABLES = [
    ("log_Ibg", "log Ibg", "log_Ibg"),
    ("log_C","log C","log_C"),
    ("A","A1","A_1"),
    ("Rm","Rm1","Rm_1"),
    ("Sigma","Sigma1","Sigma_1"),
    ("kellipsoid","k1","kellipsoid_1"),
    ("kshell","k1","kshell_1"),
    ("mu","mu1","mu_1"),

    ("A","A2","A_2"),
    ("Rm","Rm2","Rm_2"),
    ("Sigma","Sigma2","Sigma_2"),
    ("kellipsoid","k2","kellipsoid_2"),
    ("kshell","k2","kshell_2"),
    ("mu","mu2","mu_2"),

#    ("log_Ibg", "log Ibg", "log_Ibg_mag"),
#    ("log_C","log C","log_C_mag"),
#    ("A","A1","A_1_mag"),
#    ("Rm","Rm1","Rm_1_mag"),
#    ("Sigma","Sigma1","Sigma_1_mag"),
#    ("kellipsoid","k1","kellipsoid_1_mag"),
#    ("kshell","k1","kshell_1_mag"),
#    ("mu","mu1","mu_1_mag"),
#
#    ("A","A2","A_2_mag"),
#    ("Rm","Rm2","Rm_2_mag"),
#    ("Sigma","Sigma2","Sigma_2_mag"),
#    ("kellipsoid","k2","kellipsoid_2_mag"),
#    ("kshell","k2","kshell_2_mag"),
#    ("mu","mu2","mu_2_mag")    
]