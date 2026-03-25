
async def get_timetable()-> pandas.DataFrame:
    """
    Get current schedule 

    Returns
    -------
    str
        A string containing the timetable
    """
    import pandas as pd

    df = pd.read_excel("test.xlsx", sheet_name="Sheet1")
    return (df)


















async def get_additive_manufacturing_equipment(equipment_type: str=None)-> dict:
    """
    Retrieve additive manufacturing equipment and their descriptions based on type.
    Parameters
    ----------
    equipment_type : str, optional
        The category of equipment to retrieve. Valid options are:
        - 'printers' : 3D printing machines
        - 'design' : CAD and slicing software
        - 'post_processing' : Post-processing tools and equipment
        - 'quality' : Measurement and quality control tools
        - None : Returns all equipment categories (default)
    
    Returns
    -------
    dict
        A dictionary containing lists of equipment with their names and descriptions.
        Structure:
        {
            'printers': [{'name': str, 'description': str}, ...],
            'design': [{'name': str, 'description': str}, ...],
            'post_processing': [{'name': str, 'description': str}, ...],
            'quality': [{'name': str, 'description': str}, ...]
        }
    
    Raises
    ------
    ValueError
        If an invalid equipment_type is provided.
    """
    
    equipment_database = {
        'printers': [
            {
                'name': 'Fused Deposition Modeling (FDM) 3D Printers',
                'description': 'Extrude thermoplastics like PLA, ABS, and PETG layer by layer. Most common and affordable technology for prototyping and hobbyist use.'
            },
            {
                'name': 'Stereolithography (SLA) 3D Printers',
                'description': 'Use UV laser to cure liquid resin into solid parts. Produces high-resolution prints with smooth surface finish, ideal for detailed prototypes.'
            },
            {
                'name': 'Selective Laser Sintering (SLS) Machines',
                'description': 'Use laser to fuse nylon or metal powder particles. Industrial-grade technology for functional prototypes and end-use parts without supports.'
            },
            {
                'name': 'Metal 3D Printers (DMLS/SLM)',
                'description': 'Direct Metal Laser Sintering or Selective Laser Melting. Industrial machines for manufacturing complex metal components with high strength.'
            }
        ],
        
        'design': [
            {
                'name': 'CAD Software - SolidWorks',
                'description': 'Professional parametric 3D modeling software for engineering design and product development.'
            },
            {
                'name': 'CAD Software - Fusion 360',
                'description': 'Cloud-based 3D CAD/CAM platform for product design and manufacturing.'
            },
            {
                'name': 'CAD Software - AutoCAD',
                'description': 'Industry-standard 2D and 3D CAD software for precise drafting and modeling.'
            },
            {
                'name': 'Slicing Software - Cura',
                'description': 'Open-source slicing software that converts 3D models into printer instructions (G-code).'
            },
            {
                'name': 'Slicing Software - PrusaSlicer',
                'description': 'Advanced slicing software with built-in profiles for Prusa printers and generic machines.'
            },
            {
                'name': 'Slicing Software - PreForm',
                'description': 'Formlabs proprietary software for preparing and optimizing SLA resin prints.'
            }
        ],
        
        'post_processing': [
            {
                'name': 'Support Removal Tools',
                'description': 'Pliers, cutters, and flush snips for removing support structures from printed parts.'
            },
            {
                'name': 'Sanding & Polishing Tools',
                'description': 'Sandpaper, files, and polishing compounds for surface finishing and smoothing layer lines.'
            },
            {
                'name': 'UV Curing Stations',
                'description': 'Post-curing equipment for resin prints to achieve maximum strength and stability.'
            },
            {
                'name': 'Paints and Coating Equipment',
                'description': 'Airbrushes, spray guns, and finishing materials for final surface treatment and aesthetics.'
            }
        ],
        
        'quality': [
            {
                'name': 'Calipers',
                'description': 'Digital or vernier calipers for measuring dimensional accuracy of printed parts.'
            },
            {
                'name': 'Micrometers',
                'description': 'Precision measurement tools for high-accuracy thickness and diameter measurements.'
            },
            {
                'name': '3D Scanners',
                'description': 'Optical or laser scanners for verifying geometry and reverse engineering existing parts.'
            },
            {
                'name': 'Surface Roughness Testers',
                'description': 'Profilometers and roughness gauges for measuring surface finish quality on functional prototypes.'
            }
        ]
    }
    
    # Validate input and return appropriate data
    valid_types = ['printers', 'design', 'post_processing', 'quality']
    
    if equipment_type is None:
        return equipment_database
    elif equipment_type in valid_types:
        return {equipment_type: equipment_database[equipment_type]}
    else:
        raise ValueError(f"Invalid equipment_type. Must be one of: {', '.join(valid_types)} or None")

