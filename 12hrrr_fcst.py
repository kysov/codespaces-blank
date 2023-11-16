import s3fs
import numpy as np
from datetime import datetime, timedelta, date
import numcodecs as ncd
import metpy
import metpy.units as mu
import metpy.calc as mc
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from pytz import timezone  # New import for timezone conversion

def get_nearest_point(projection, chunk_index, longitude, latitude):
    x, y = projection.transform_point(longitude, latitude, ccrs.PlateCarree())
    return chunk_index.sel(x=x, y=y, method="nearest")

def retrieve_data(s3_url):
    with fs.open(s3_url, 'rb') as compressed_data:
        buffer = ncd.blosc.decompress(compressed_data.read())

        dtype = "<f2"
        if "surface/PRES" in s3_url:
            dtype = "<f4"
        
        chunk = np.frombuffer(buffer, dtype=dtype)
        
        entry_size = 150*150
        num_entries = len(chunk)//entry_size
        
        if num_entries == 1:
            data_array = np.reshape(chunk, (150, 150))
        else:
            data_array = np.reshape(chunk, (num_entries, 150, 150))
    
    return data_array

# Lists
forecast_hour = []
date_list = []
temp_list = []
pres_list = []
snow_list = []
gust_list = []
accum_snow = []
rh_list = []
ugrd_list = []
vgrd_list = []
wind_list = []
carinal_wind_dir = []

surface_list = ["SNOD", "GUST", "ASNOW_acc_fcst", "PRES","TMP"]
twom_list = ["RH"]
tenm_list = ["UGRD", "VGRD", "WIND_1hr_max_fcst"]

# Location
point_lat = 39.58148838130895
point_lon = -105.94259648925797

# Find chunk that contains coordinates
fs = s3fs.S3FileSystem(anon=True)

chunk_index = xr.open_zarr(s3fs.S3Map("s3://hrrrzarr/grid/HRRR_chunk_index.zarr", s3=fs))
projection = ccrs.LambertConformal(central_longitude=262.5, 
                                   central_latitude=38.5, 
                                   standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                                     semiminor_axis=6371229))
nearest_point = get_nearest_point(projection, chunk_index, point_lon, point_lat)
fcst_chunk_id = f"0.{nearest_point.chunk_id.values}"

# Date range:
now = datetime.now() - timedelta(hours=3)
day = now.strftime("%Y%m%d")
hr = now.strftime("%H")

for var in surface_list:
    level = 'surface'
    data_url = f'hrrrzarr/sfc/{day}/{day}_{hr}z_fcst.zarr/{level}/{var}/{level}/{var}/'
    
    data = retrieve_data(data_url + fcst_chunk_id)
    gridpoint_forecast = data[:, nearest_point.in_chunk_y, nearest_point.in_chunk_x]
    
    for n in range(3, 15):
        if var == "PRES":
            pres_list.append(gridpoint_forecast[n])
            date_list.append(day)
            forecast_hour.append(n + int(hr))
        if var == "TMP":
            temp_list.append(gridpoint_forecast[n])
        elif var == "SNOD":
            snow_list.append(gridpoint_forecast[n])
        elif var == "GUST":
            gust_list.append(gridpoint_forecast[n])
        elif var == "ASNOW_acc_fcst":
            accum_snow.append(gridpoint_forecast[n])

for var in twom_list:
    level = '2m_above_ground'
    data_url = f'hrrrzarr/sfc/{day}/{day}_{hr}z_fcst.zarr/{level}/{var}/{level}/{var}/'
    
    data = retrieve_data(data_url + fcst_chunk_id)
    gridpoint_forecast = data[:, nearest_point.in_chunk_y, nearest_point.in_chunk_x]
    
    for n in range(3, 15):
        if var == "RH":
            rh_list.append(gridpoint_forecast[n])

for var in tenm_list:
    level = '10m_above_ground'
    data_url = f'hrrrzarr/sfc/{day}/{day}_{hr}z_fcst.zarr/{level}/{var}/{level}/{var}/'
    
    data = retrieve_data(data_url + fcst_chunk_id)
    gridpoint_forecast = data[:, nearest_point.in_chunk_y, nearest_point.in_chunk_x]
    
    for n in range(3, 15):
        if var == "UGRD":
            ugrd_list.append(gridpoint_forecast[n])
        elif var == "VGRD":
            vgrd_list.append(gridpoint_forecast[n])
        elif var == "WIND_1hr_max_fcst":
            wind_list.append(gridpoint_forecast[n])

# Calc wind direction
for i in range(0, len(ugrd_list)):
    direction = metpy.calc.wind_direction(ugrd_list[i]*mu.units.metre/mu.units.second, vgrd_list[i]*mu.units.metre/mu.units.second)
    cardinal_direction = direction.magnitude
    carinal_wind_dir.append(mc.angle_to_direction(int(cardinal_direction)))

# Put output into a dataframe
output = pd.DataFrame({
    'Date': date_list,
    'Hour_UTC': forecast_hour,
    'Pres': pres_list,
    'Temp': temp_list,
    'Snow_Depth': snow_list,
    'Hourly_Snow': accum_snow,
    'RH': rh_list,
    'Gust': gust_list,
    'Wind_Dir': carinal_wind_dir,
    'Wind_Speed': wind_list,
})

# Convert 'Date' to datetime format
output['Date'] = pd.to_datetime(output['Date'], format='%Y%m%d')

# Correct hours that are 24 or more and adjust dates accordingly
for i in range(len(output)):
    if output.at[i, 'Hour_UTC'] >= 24:
        extra_days = int(output.at[i, 'Hour_UTC'] // 24)  # Convert to native Python int
        output.at[i, 'Date'] += timedelta(days=extra_days)
        output.at[i, 'Hour_UTC'] %= 24

# New timezone conversion code
denver_timezone = timezone('America/Denver')  # Define the timezone for Denver
output['Datetime_UTC'] = pd.to_datetime(output['Date'].astype(str) + ' ' + output['Hour_UTC'].astype(str).str.zfill(2))
output['Datetime_UTC'] = output['Datetime_UTC'].dt.tz_localize('UTC')  # Localize as UTC
output['Datetime_Local'] = output['Datetime_UTC'].dt.tz_convert(denver_timezone)  # Convert to Denver Time
output.drop(columns=['Date', 'Hour_UTC'], inplace=True)  # Drop original Date and Hour_UTC columns
output['Date_Local'] = output['Datetime_Local'].dt.strftime('%m/%d/%y')  # Extract local date
output['Hour_Local'] = output['Datetime_Local'].dt.hour  # Extract local hour

output.Pres = output.Pres / 3386  # inHg
output.Temp = (output.Temp - 273.15) * (9/5) + 32  # F
output.Snow_Depth = output.Snow_Depth * 39.37  # in
output.Hourly_Snow = output.Hourly_Snow * 39.37  # in
output.Gust = output.Gust * 2.24  # mph

# Define the desired order of columns
column_order = ['Date_Local', 'Hour_Local', 'Temp', 'Snow_Depth', 'Hourly_Snow', 'RH', 'Gust', 'Wind_Dir', 'Wind_Speed', 'Pres', 'Datetime_UTC']

# Reorder the columns in the DataFrame
output = output[column_order]

# original print format
#print(output)

# Put output into CSV file
output.to_csv('12hr_hrrr_fcst.csv', index=False)

# Convert the DataFrame to a CSV string and print
csv_string = output.to_csv(index=False)
print(csv_string)