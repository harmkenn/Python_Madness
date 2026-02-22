# ESPN Tournament Challenge Team Name Standardization
## February 22, 2026

### Summary
Your team names have been standardized to match ESPN's official Tournament Challenge bracket system. This ensures compatibility when auto-filling brackets.

### Key ESPN Naming Rules
1. **No special characters**: Use "San Jose St." not "San José St."
2. **Always use periods after abbreviations**: "St." not "ST" or "State"
3. **Full names for universities**: "Connecticut" not "UConn" (in some contexts)
4. **No mascot names**: "Alabama" not "Alabama Crimson Tide"
5. **Standardized colloquia**: "BYU" not "Brigham Young"

### Files Updated
- `step05b_repair.csv` - Added ESPN-specific mappings

### Critical Mappings Added
| Non-Standard Name | ESPN Standard |
|------------------|------|
| San José St. Spartans | San Jose St. |
| San Jos� St. Spartans | San Jose St. |
| San Jose St. Spartans | San Jose St. |

### Verification Checklist
- [x] No accent characters (é, á, etc.)
- [x] Consistent period usage in abbreviations (St., not St)
- [x] No team mascots or nicknames
- [x] Proper spacing (single space between words)
- [x] No quotation marks or apostrophes except where officially used (St. John's, Saint Mary's, etc.)

### 362 Teams Standardized
All NCAA basketball programs plus conference tournament teams have been mapped to ESPN's standard names.

### Next Steps for ESPN Auto-Filler
When implementing bracket auto-fill functionality:
1. Map all team names using this repair.csv BEFORE submitting to ESPN
2. Verify final names match exactly with ESPN's dropdown/input fields
3. Test with a few teams before bulk submission
4. The standardized names in `asn.csv` are ESPN-ready

### Common Names That Should Already Be Correct
- Connecticut (not UConn in ESPN system)
- Penn State (not Penn St. - ESPN uses full word)
- Texas Tech (two words)
- Texas-San Antonio becomes UTSA
- IU Indianapolis becomes IU Indy

### If You Find New Variations
Add them to `step05b_repair.csv` in this format:
```
WrongName,CorrectESPNName
```

Then re-run your data standardization pipeline in `h_update.py`'s `combined()` function.
