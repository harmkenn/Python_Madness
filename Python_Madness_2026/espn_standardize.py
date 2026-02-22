"""
ESPN Team Name Standardization Utility
=====================================
This script creates and validates ESPN tournament challenge compatible team names.
ESPN uses a specific format for all 68+ teams in March Madness.
"""

import pandas as pd
import os

# OFFICIAL ESPN TOURNAMENT CHALLENGE TEAM NAMES (2024-2026)
# These are the exact names ESPN uses in their bracket system
ESPN_OFFICIAL_NAMES = {
    'Abilene Christian', 'Air Force', 'Akron', 'Alabama', 'Alabama A&M', 'Alabama St.',
    'Albany', 'Alcorn St.', 'American', 'Appalachian St.', 'Arizona', 'Arizona St.',
    'Arkansas', 'Arkansas Pine Bluff', 'Arkansas St.', 'Army', 'Auburn', 'Austin Peay',
    'BYU', 'Ball St.', 'Baylor', 'Bellarmine', 'Belmont', 'Bethune Cookman',
    'Binghamton', 'Boise St.', 'Boston College', 'Boston University', 'Bowling Green',
    'Bradley', 'Brown', 'Bryant', 'Bucknell', 'Buffalo', 'Butler',
    'CSUN', 'Cal Baptist', 'Cal Poly', 'Cal St. Bakersfield', 'Cal St. Fullerton',
    'California', 'Campbell', 'Canisius', 'Central Arkansas', 'Central Connecticut',
    'Central Michigan', 'Charleston', 'Charleston Southern', 'Charlotte', 'Chattanooga',
    'Chicago St.', 'Cincinnati', 'Clemson', 'Cleveland St.', 'Coastal Carolina',
    'Colgate', 'Colorado', 'Colorado St.', 'Columbia', 'Connecticut', 'Coppin St.',
    'Cornell', 'Creighton',
    'Dartmouth', 'Davidson', 'Dayton', 'DePaul', 'Delaware', 'Delaware St.',
    'Denver', 'Detroit Mercy', 'Drake', 'Drexel', 'Duke', 'Duquesne',
    'East Carolina', 'East Tennessee St.', 'East Texas A&M', 'Eastern Illinois',
    'Eastern Kentucky', 'Eastern Michigan', 'Eastern Washington', 'Elon', 'Evansville',
    'FIU', 'Fairfield', 'Fairleigh Dickinson', 'Florida', 'Florida A&M',
    'Florida Atlantic', 'Florida Gulf Coast', 'Florida St.', 'Fordham', 'Fresno St.',
    'Furman',
    'Gardner Webb', 'George Mason', 'George Washington', 'Georgetown', 'Georgia',
    'Georgia Southern', 'Georgia St.', 'Georgia Tech', 'Gonzaga', 'Grambling St.',
    'Grand Canyon', 'Green Bay',
    'Hampton', 'Hawaii', 'High Point', 'Hofstra', 'Holy Cross', 'Houston',
    'Houston Christian', 'Howard',
    'IU Indy', 'Idaho', 'Idaho St.', 'Illinois', 'Illinois Chicago', 'Illinois St.',
    'Incarnate Word', 'Indiana', 'Indiana St.', 'Iona', 'Iowa', 'Iowa St.',
    'Jackson St.', 'Jacksonville', 'Jacksonville St.', 'James Madison', 'Kansas',
    'Kansas St.', 'Kennesaw St.', 'Kent St.', 'Kentucky',
    'La Salle', 'Lafayette', 'Lamar', 'Le Moyne', 'Lehigh', 'Liberty',
    'Lindenwood', 'Lipscomb', 'Little Rock', 'Long Beach St.', 'Longwood',
    'Louisiana', 'Louisiana Monroe', 'Louisiana Tech', 'Louisville',
    'Loyola Chicago', 'Loyola MD', 'Loyola Marymount', 'LSU',
    'LIU', 'Maine', 'Manhattan', 'Marist', 'Marquette', 'Marshall',
    'Maryland', 'Maryland Eastern Shore', 'Massachusetts', 'McNeese St.',
    'Memphis', 'Mercer', 'Mercyhurst', 'Merrimack', 'Miami FL', 'Miami OH',
    'Michigan', 'Michigan St.', 'Middle Tennessee', 'Milwaukee', 'Minnesota',
    'Mississippi', 'Mississippi St.', 'Mississippi Valley St.', 'Missouri',
    'Missouri St.', 'Monmouth', 'Montana', 'Montana St.', 'Morehead St.',
    'Morgan St.', 'Mount St. Mary\'s', 'Murray St.',
    'NJIT', 'Navy', 'Nebraska', 'Nebraska Omaha', 'Nevada', 'New Hampshire',
    'New Mexico', 'New Mexico St.', 'New Orleans', 'Niagara', 'Nicholls St.',
    'Norfolk St.', 'North Alabama', 'North Carolina', 'North Carolina A&T',
    'North Carolina Central', 'North Carolina St.', 'North Dakota', 'North Dakota St.',
    'North Florida', 'North Texas', 'Northeastern', 'Northern Arizona',
    'Northern Colorado', 'Northern Illinois', 'Northern Iowa', 'Northern Kentucky',
    'Northwestern', 'Northwestern St.', 'Notre Dame',
    'Oakland', 'Ohio', 'Ohio St.', 'Oklahoma', 'Oklahoma St.', 'Old Dominion',
    'Oral Roberts', 'Oregon', 'Oregon St.',
    'Pacific', 'Penn', 'Penn St.', 'Pepperdine', 'Pittsburgh', 'Portland',
    'Portland St.', 'Prairie View A&M', 'Presbyterian', 'Princeton', 'Providence',
    'Purdue', 'Purdue Fort Wayne',
    'Queens', 'Quinnipiac',
    'Radford', 'Rhode Island', 'Rice', 'Richmond', 'Rider', 'Robert Morris',
    'Rutgers',
    'SMU', 'Sacramento St.', 'Sacred Heart', 'Saint Francis', 'Saint Joseph\'s',
    'Saint Louis', 'Saint Mary\'s', 'Saint Peter\'s', 'Sam Houston St.', 'Samford',
    'San Diego', 'San Diego St.', 'San Francisco', 'San Jose St.', 'Santa Clara',
    'Seattle', 'Seton Hall', 'Siena', 'SIUE', 'South Alabama', 'South Carolina',
    'South Carolina St.', 'South Dakota', 'South Dakota St.', 'South Florida',
    'Southeast Missouri', 'Southeastern Louisiana', 'Southern', 'Southern Illinois',
    'Southern Indiana', 'Southern Miss', 'Southern Utah', 'St. Bonaventure',
    'St. John\'s', 'St. Thomas', 'Stanford', 'Stephen F. Austin', 'Stetson',
    'Stonehill', 'Stony Brook', 'Syracuse',
    'TCU', 'Tarleton St.', 'Temple', 'Tennessee', 'Tennessee Martin',
    'Tennessee St.', 'Tennessee Tech', 'Texas', 'Texas A&M',
    'Texas A&M Corpus Chris', 'Texas Southern', 'Texas St.', 'Texas Tech',
    'The Citadel', 'Toledo', 'Towson', 'Troy', 'Tulane', 'Tulsa',
    'UAB', 'UC Davis', 'UC Irvine', 'UC Riverside', 'UC San Diego',
    'UC Santa Barbara', 'UCF', 'UCLA', 'UMBC', 'UMKC', 'UMass Lowell',
    'UNC Asheville', 'UNC Greensboro', 'UNC Wilmington', 'UNLV', 'USC',
    'UT Arlington', 'UT Rio Grande Valley', 'UTEP', 'UTSA', 'Utah',
    'Utah St.', 'Utah Tech', 'Utah Valley',
    'VCU', 'VMI', 'Valparaiso', 'Vanderbilt', 'Vermont', 'Villanova',
    'Virginia', 'Virginia Tech',
    'Wagner', 'Wake Forest', 'Washington', 'Washington St.', 'Weber St.',
    'West Georgia', 'West Virginia', 'Western Carolina', 'Western Illinois',
    'Western Kentucky', 'Western Michigan', 'Wichita St.', 'William & Mary',
    'Winthrop', 'Wisconsin', 'Wofford', 'Wright St.', 'Wyoming',
    'Xavier', 'Yale', 'Youngstown St.'
}

def analyze_team_names():
    """
    Analyze current team names in your data and check against ESPN official names.
    """
    base_path = "Python_Madness_2026/data"
    
    # Load your current standard names
    asn_path = os.path.join(base_path, "asn.csv")
    repair_path = os.path.join(base_path, "step05b_repair.csv")
    
    if os.path.exists(asn_path):
        asn = pd.read_csv(asn_path)
        current_names = set(asn.iloc[:, 0].dropna().unique())
    else:
        current_names = set()
    
    # Check for discrepancies
    missing_from_espn = current_names - ESPN_OFFICIAL_NAMES
    
    print("=" * 60)
    print("ESPN TEAM NAME STANDARDIZATION ANALYSIS")
    print("=" * 60)
    print(f"\nTotal official ESPN names: {len(ESPN_OFFICIAL_NAMES)}")
    print(f"Your current standard names: {len(current_names)}")
    print(f"\nNames in your data NOT in ESPN's official list: {len(missing_from_espn)}")
    
    if missing_from_espn:
        print("\nNames to investigate/fix:")
        for name in sorted(missing_from_espn):
            print(f"  - {name}")
    
    # Load repair CSV to show current mappings
    if os.path.exists(repair_path):
        repair = pd.read_csv(repair_path)
        print(f"\nCurrent repair mappings: {len(repair)} entries")
    
    return {
        'official_names': ESPN_OFFICIAL_NAMES,
        'current_names': current_names,
        'missing_from_espn': missing_from_espn
    }

if __name__ == "__main__":
    result = analyze_team_names()
