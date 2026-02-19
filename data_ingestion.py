import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class SpectrumObject:
    id: str
    x: np.ndarray
    y: np.ndarray

class UniversalParser:
    def parse_pahdb_xml(self, filepath: str, limit=None):
        """Iteratively parses the theoretical PAHdb XML file."""
        spectra = []
        try:
            print(f"Loading PAHdb: {filepath}...")
            context = ET.iterparse(filepath, events=('end',))
            
            for event, elem in context:
                if elem.tag.endswith('}specie') or elem.tag == 'specie':
                    uid = elem.get('uid')
                    x_vals, y_vals = [], []
                    
                    transitions = None
                    for child in elem:
                        if child.tag.split('}')[-1] == 'transitions':  
                            transitions = child
                            break
                    
                    if transitions is not None:
                        for mode in transitions:
                            if mode.tag.split('}')[-1] == 'mode':
                                freq, intens = None, None
                                for prop in mode:
                                    tag = prop.tag.split('}')[-1]
                                    if tag == 'frequency': freq = prop.text
                                    elif tag == 'intensity': intens = prop.text
                                
                                if freq and intens:
                                    x_vals.append(float(freq))
                                    y_vals.append(float(intens))

                    if x_vals and len(x_vals) > 0:
                        idx = np.argsort(x_vals)
                        spectra.append(SpectrumObject(
                            id=f"PAH_{uid}",
                            x=np.array(x_vals)[idx],
                            y=np.array(y_vals)[idx]
                        ))
                    
                    elem.clear() # Free memory
                    if len(spectra) % 100 == 0:
                        print(f"Loaded {len(spectra)} species...", end='\r')
                    if limit and len(spectra) >= limit:
                        break
                        
            print(f"Successfully loaded {len(spectra)} PAH standards.")
            return spectra
            
        except Exception as e:
            print(f"Error parsing PAHdb XML: {e}")
            return []
