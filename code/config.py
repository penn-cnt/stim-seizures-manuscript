from os.path import join as ospj
from os.path import exists as ospe
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# suppress setting with copy warning
pd.options.mode.chained_assignment = None

# Set up matplotlib parameters automatically when module is imported
plt.rcParams['image.cmap'] = 'magma'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1

class Config:
    # Hardcoded paths
    usr = "wojemann" # change to your username
    passpath = "/Users/wojemann/Documents/CNT/woj_ieeglogin.bin" # change to your login path
    datapath = "." # change to your raw data path
    prodatapath = "/Users/wojemann/Documents/CNT/stim_seizures_data/PROCESSED_DATA" # change to your processed data path
    metapath = "/Users/wojemann/Documents/CNT/stim_seizures_data/METADATA" # change to your metadata path
    figpath = "/Users/wojemann/Documents/CNT/stim_seizures_data/figures" # change to your figure path
    
    # Patient data
    _patients = [
        {"ptID": "HUP224", "ieeg_ids": ["HUP224_phaseII","HUP224_CCEP"], "interictal_training": ["HUP224_phaseII",5915]},
        {"ptID": "HUP225", "ieeg_ids": ["HUP225_phaseII","HUP225_CCEP"], "interictal_training": ["HUP225_phaseII",71207]},
        {"ptID": "HUP229", "ieeg_ids": ["HUP229_phaseII","HUP229_CCEP"], "interictal_training": ["HUP229_phaseII",149146]},
        {"ptID": "HUP230", "ieeg_ids": ["HUP230_phaseII","HUP230_CCEP"], "interictal_training":["HUP230_phaseII",25350]},
        {"ptID": "HUP235", "ieeg_ids": ["HUP235_phaseII","HUP235_CCEP"], "interictal_training": ["HUP235_phaseII",307651]},
        {"ptID": "HUP238", "ieeg_ids": ["HUP238_phaseII","HUP238_CCEP"], "interictal_training": ["HUP238_phaseII",100011]},
        {"ptID": "HUP246", "ieeg_ids": ["HUP246_phaseII","HUP246_CCEP"], "interictal_training": ["HUP246_phaseII",100000]},
        {"ptID": "HUP247", "ieeg_ids": ["HUP247_phaseII","HUP247_CCEP"], "interictal_training": ["HUP247_phaseII",17590]},
        {"ptID": "HUP249", "ieeg_ids": ["HUP249_phaseII","HUP249_CCEP"], "interictal_training": ["HUP249_phaseII",24112]},
        {"ptID": "HUP250", "ieeg_ids": ["HUP250_phaseII","HUP250_CCEP"], "interictal_training": ["HUP250_phaseII",24841]},
        {"ptID": "HUP253", "ieeg_ids": ["HUP253_phaseII","HUP253_CCEP"], "interictal_training": ["HUP253_phaseII",77624]},
        {"ptID": "HUP257", "ieeg_ids": ["HUP257_phaseII","HUP257_CCEP"], "interictal_training": ["HUP257_phaseII",15600]},
        {"ptID": "HUP261", "ieeg_ids": ["HUP261_phaseII","HUP261_CCEP"], "interictal_training": ["HUP261_phaseII",3356.07]},
        {"ptID": "HUP263", "ieeg_ids": ["HUP263_phaseII","HUP263_CCEP"], "interictal_training": ["HUP263_phaseII",28040]},
        {"ptID": "HUP266", "ieeg_ids": ["HUP266_phaseII","HUP266_CCEP"], "interictal_training": ["HUP266_phaseII",25165]},
        {"ptID": "HUP267", "ieeg_ids": ["HUP267_phaseII","HUP267b_phaseII","HUP267_CCEP"], "interictal_training": ["HUP267_phaseII", 58083.49]},
        {"ptID": "HUP273", "ieeg_ids": ["HUP273_phaseII","HUP273b_phaseII","HUP273c_phaseII","HUP273_CCEP"], "interictal_training": ["HUP273b_phaseII", 41715]},
        {"ptID": "HUP275", "ieeg_ids": ["HUP275_CCEP","HUP275_phaseII"], "interictal_training": ["HUP275_phaseII", 310639]},
        {"ptID": "HUP288", "ieeg_ids": ["HUP288_CCEP","HUP288_phaseII"], "interictal_training": ["HUP288_phaseII", 25015]},
        {"ptID": "CHOP005", "ieeg_ids": ["CHOPCCEP_005","CHOP005"], "interictal_training": ["CHOP005",14190.17]},
        {"ptID": "CHOP010", "ieeg_ids": ["CHOPCCEP_010","CHOP010a","CHOP010b","CHOP010c"], "interictal_training": ["CHOP010a",10845.95]},
        {"ptID": "CHOP015", "ieeg_ids": ["CHOPCCEP_015"], "interictal_training": []},
        {"ptID": "CHOP024", "ieeg_ids": ["CHOPCCEP_024","CHOP024"], "interictal_training": ["CHOP024",112138.27]},
        {"ptID": "CHOP026", "ieeg_ids": ["CHOPCCEP_026","CHOP026"], "interictal_training": ["CHOP026",76411.33]},
        {"ptID": "CHOP028", "ieeg_ids": ["CHOPCCEP_028","CHOP028"], "interictal_training": ["CHOP028",7517.56]},
        {"ptID": "CHOP035", "ieeg_ids": ["CHOPCCEP_035","CHOP035"], "interictal_training": ["CHOP035",82282.00]},
        {"ptID": "CHOP037", "ieeg_ids": ["CHOPCCEP_037","CHOP037"], "interictal_training": ["CHOP037",58173.01]},
        {"ptID": "CHOP038", "ieeg_ids": ["CHOPCCEP_038","CHOP038"], "interictal_training":[]},
        {"ptID": "CHOP041", "ieeg_ids": ["CHOPCCEP_041","CHOP041"], "interictal_training": ["CHOP041",112959.70]},
        {"ptID": "CHOP044", "ieeg_ids": ["CHOPCCEP_044","CHOP044"], "interictal_training": ["CHOP044",4070.79]},
        {"ptID": "CHOP045", "ieeg_ids": ["CHOPCCEP_045","CHOP045"], "interictal_training": ["CHOP045",13156.41]},
        {"ptID": "CHOP046", "ieeg_ids": ["CHOPCCEP_046","CHOP046"], "interictal_training": []},
        {"ptID": "CHOP049", "ieeg_ids": ["CHOPCCEP_049","CHOP049"], "interictal_training": ["CHOP049",8313.95]}
    ]
    
    @classmethod
    def create_r_config(cls, output_path=None):
        """
        Create a JSON configuration file that can be easily read by R scripts.
        
        Args:
            output_path (str): Path where to save the JSON config file. 
                             If None, saves to metapath/r_config.json
                             
        Returns:
            str: Path where the config file was saved
        """
        # Create the configuration dictionary
        config_dict = {
            "prodatapath": cls.prodatapath,
            "metapath": cls.metapath,
            "figpath": cls.figpath,
            "usr": cls.usr,
            "datapath": cls.datapath,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        # Determine output path
        if output_path is None:
            # Create metapath directory if it doesn't exist
            os.makedirs(cls.metapath, exist_ok=True)
            output_path = ospj(cls.metapath, "r_config.json")
        
        # Write JSON file
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"R configuration file created at: {output_path}")
        print("Contents:")
        for key, value in config_dict.items():
            if key != "created_at":
                print(f"  {key}: {value}")
        
        return output_path
    
    @classmethod
    def deal(cls, attrs=None, flag=None):
        """
        Return all configuration data similar to load_config function.
        
        Args:
            flag (str): Patient cohort filter ('HUP' or 'CHOP'). Defaults to None.
            attrs (iterable): List/tuple of attribute names to return. If None, returns all.
                             Available: 'usr', 'passpath', 'datapath', 'prodatapath', 
                             'metapath', 'figpath', 'patient_table', 'rid_hup', 'pt_list'
            
        Returns:
            tuple: If attrs is None: (usr, passpath, datapath, prodatapath, metapath, figpath, 
                                    patient_table, rid_hup, pt_list)
                   If attrs provided: tuple of requested attributes in specified order
                   
        Examples:
            # Get all attributes
            usr, passpath, datapath, prodatapath, metapath, figpath, patient_table, rid_hup, pt_list = Config.deal()
            
            # Get only specific attributes
            usr, metapath, figpath = Config.deal(attrs=['usr', 'metapath', 'figpath'])
            datapath, patient_table = Config.deal('HUP', ['datapath', 'patient_table'])
        """
        # Process patient data
        patient_table = pd.DataFrame(cls._patients).sort_values('ptID').reset_index(drop=True)
        if flag == 'HUP':
            patient_table = patient_table[patient_table.ptID.apply(lambda x: x[:3]) == 'HUP']
        elif flag == 'CHOP':
            patient_table = patient_table[patient_table.ptID.apply(lambda x: x[:4]) == 'CHOP']
        
        # Load RID-HUP mapping
        if ospe(ospj(cls.metapath, 'rid_hup.csv')):
            rid_hup = pd.read_csv(ospj(cls.metapath, 'rid_hup.csv'))
        else:
            rid_hup = pd.DataFrame(columns=['hupsubjno','record_id'])
        pt_list = patient_table.ptID.to_numpy()
        
        # Create mapping of all available attributes
        all_attrs = {
            'usr': cls.usr,
            'passpath': cls.passpath,
            'datapath': cls.datapath,
            'prodatapath': cls.prodatapath,
            'metapath': cls.metapath,
            'figpath': cls.figpath,
            'patient_table': patient_table,
            'rid_hup': rid_hup,
            'pt_list': pt_list
        }
        
        # If specific attributes requested, return only those
        if attrs is not None:
            try:
                # Handle both strings and iterables
                if isinstance(attrs, str):
                    attrs = [attrs]
                
                # Validate all requested attributes exist
                invalid_attrs = [attr for attr in attrs if attr not in all_attrs]
                if invalid_attrs:
                    raise ValueError(f"Invalid attribute(s): {invalid_attrs}. "
                                   f"Available: {list(all_attrs.keys())}")
                
                # Return requested attributes in specified order
                result = tuple(all_attrs[attr] for attr in attrs)
                return result[0] if len(result) == 1 else result
                
            except (TypeError, ValueError) as e:
                raise ValueError(f"attrs must be a string or iterable of attribute names. {e}")
        
        # Default: return all attributes in original order
        return (cls.usr, cls.passpath, cls.datapath, cls.prodatapath, 
                cls.metapath, cls.figpath, patient_table, rid_hup, pt_list)
    
    def __new__(cls):
        """Prevent instantiation by raising an error."""
        raise TypeError(f"{cls.__name__} should not be instantiated. Use class attributes directly.")

# When this script is run directly, create the R configuration file
if __name__ == "__main__":
    Config.create_r_config()
    print("\nTo use this configuration in R scripts, they will automatically load from:")
    print(f"{Config.metapath}/r_config.json")
    