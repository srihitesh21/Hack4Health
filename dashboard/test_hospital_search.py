#!/usr/bin/env python3
"""
Test script for hospital search functionality
"""

import requests
import json
import time

def test_hospital_search():
    """Test the hospital search functionality"""
    
    # Test addresses
    test_addresses = [
        "123 Main St, Vancouver, BC",
        "Stanley Park, Vancouver, BC", 
        "UBC Hospital, Vancouver, BC",
        "Vancouver General Hospital, Vancouver, BC"
    ]
    
    print("🏥 Testing Hospital Search Functionality")
    print("=" * 50)
    
    for i, address in enumerate(test_addresses, 1):
        print(f"\n{i}. Testing address: {address}")
        
        try:
            # Test geocoding
            geocode_url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'q': address,
                'format': 'json',
                'limit': 1
            }
            
            response = requests.get(geocode_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    lat = float(data[0]['lat'])
                    lon = float(data[0]['lon'])
                    print(f"   ✅ Geocoding successful: {lat:.6f}, {lon:.6f}")
                    
                    # Test hospital search
                    hospital_url = "https://overpass-api.de/api/interpreter"
                    query = f"""
                    [out:json][timeout:25];
                    (
                      node["amenity"="hospital"](around:5000,{lat},{lon});
                      way["amenity"="hospital"](around:5000,{lat},{lon});
                      relation["amenity"="hospital"](around:5000,{lat},{lon});
                    );
                    out body;
                    >;
                    out skel qt;
                    """
                    
                    hospital_response = requests.post(hospital_url, data=query, timeout=15)
                    if hospital_response.status_code == 200:
                        hospital_data = hospital_response.json()
                        if hospital_data.get('elements'):
                            hospitals = [elem for elem in hospital_data['elements'] if elem.get('tags', {}).get('name')]
                            print(f"   ✅ Found {len(hospitals)} hospitals nearby")
                            if hospitals:
                                closest = hospitals[0]
                                name = closest.get('tags', {}).get('name', 'Unknown Hospital')
                                print(f"   📍 Closest: {name}")
                        else:
                            print(f"   ⚠️  No hospitals found in database")
                    else:
                        print(f"   ❌ Hospital search failed: {hospital_response.status_code}")
                else:
                    print(f"   ❌ Geocoding failed: No results found")
            else:
                print(f"   ❌ Geocoding failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        time.sleep(1)  # Be nice to the APIs
    
    print("\n" + "=" * 50)
    print("✅ Hospital search test completed!")
    print("\nTo test the full dashboard:")
    print("1. Open http://localhost:5003 in your browser")
    print("2. Scroll down to the Emergency Services section")
    print("3. Type an address in the input field")
    print("4. Click 'Find Hospital' or press Enter")
    print("5. Check the browser console for any errors")

if __name__ == "__main__":
    test_hospital_search() 