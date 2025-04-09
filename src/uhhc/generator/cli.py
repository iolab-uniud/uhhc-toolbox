import click
from typing import Literal, TextIO
from .. import __package__ as package_name
from ._utils import (
    find_appropriate_crs, 
    download_osm_data, 
    find_osm_bounds, 
    prepare_osrm_data,
    download_population_density,
    download_gadm_administrative_data
)
from .generator import generate_spatial, generate_temporal
from .. models.instance_models import Instance, LunchBreakSpec, MetaData, GeneratorInformation, CostComponents, PatientGeneratorInformation, CaregiverGeneratorInformation
from pathlib import Path
import shapely as sp
import platformdirs
import geopandas as gpd   
from beaupy import select_multiple, select
import humanize
import autopage
import pyosrm
import re
import yaml
import random
import numpy as np
import math

@click.group()
def cli() -> None:
    """This is the main command group related to data files for geographic management and instance generation."""
    pass

# FIXME: document better
@cli.group()
def area() -> None:
    """Commands related to geographic data files management."""
    pass

@area.command(name="get")
@click.argument('city', type=str, required=True)
@click.argument('radius', type=float, required=False, default=30.0)
@click.option('--name', '-n', type=str, required=False)
@click.option('--compress', '-c', is_flag=True, default=False)
@click.option('--force', '-f', is_flag=True, default=False)
def get_area(city : str, radius : float, name : str, compress : bool, force : bool) -> None:
    radius = np.round(float(radius), 1)
    if name is None:
        name = f'{city}-{radius}km'
    data_dir = Path(platformdirs.user_data_dir(package_name))
    data_dir.mkdir(parents=True, exist_ok=True)
    # Geocode the city
    res = gpd.tools.geocode(city, provider='nominatim', user_agent=package_name)
    # The CRS of the result is WGS 84 (EPSG:4326), so we can set it explicitly
    res.set_crs(epsg=4326, inplace=True)
    click.echo(f'Getting area {name} centered at {res.iloc[0].address} ({res.iloc[0].geometry}), with a radius of {radius}km and saving into {data_dir}')

    # Find an appropriate CRS for the UTM zone
    crs = find_appropriate_crs(res.iloc[0].geometry.y, res.iloc[0].geometry.x)
    # Create a GeoDataFrame with the bounding box, with the appropriate CRS
    bbox = res.to_crs(crs).buffer(radius * 1000)
    bounds = sp.box(*bbox.to_crs(epsg=4326).total_bounds)
    if not force:
        click.echo(f'Checking if the data is already downloaded')
        for file in data_dir.iterdir():
            if '.osm' not in file.suffixes:
                continue
            file_bbox = find_osm_bounds(file)
            if file_bbox.buffer(1E-6).contains(bounds):
                click.echo(f'OpenMap data already available in {file}, skipping download')
                return
    click.echo(f'Downloading data from OpenMap for {name}')
    download_osm_data(*bbox.to_crs(epsg=4326).total_bounds, output_filename=data_dir / f'{name}.osm', bzip=compress)

    click.echo(f'OpenMap data downloaded successfully in {data_dir}')

@area.command()
def list() -> None:
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if '.osm' in d.suffixes]
    if not files:
        click.echo('No OpenMap data available')
        return

    with autopage.AutoPager() as out:
        for d in files:
            out.write(f"{d.name} ({humanize.naturalsize(d.stat().st_size, binary=True)}) {find_osm_bounds(d).bounds}\n")

@area.command()
def delete() -> None:
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if '.osm' in d.suffixes]
    if not files:
        click.echo('No OpenMap data to delete')
        return
    selected = select_multiple([f'{d.name} ({humanize.naturalsize(d.stat().st_size, binary=True)})' for d in files], return_indices=True, pagination=True)
    if selected:
        click.confirm(f'Are you sure you want to delete {len(selected)} file(s)?', abort=True)
        freed = 0
        for i in selected:
            click.echo(f'Deleting {files[i]}')
            freed += files[i].stat().st_size
            files[i].unlink()
        click.echo(f'Freed {humanize.naturalsize(freed, binary=True)}')

# FIXME: document better using pyosrm
@cli.group()
def routes() -> None:
    """Commands related to routing file management."""
    pass

@routes.command()
def create() -> None:
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if '.osm' in d.suffixes]
    if not files:
        click.echo('No OpenMap data to process')
        return
    selected = select_multiple([f'{d.name} ({humanize.naturalsize(d.stat().st_size, binary=True)})' for d in files], return_indices=True, pagination=True)
    if selected:
        click.confirm(f'Are you sure you want to process {len(selected)} file(s)?', default=True, abort=True)
        for i in selected:
            click.echo(f'Processing {files[i].name}')
            prepare_osrm_data(files[i])
        click.echo(f'Processed {len(selected)} file(s)')

@routes.command()
def list() -> None:
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if d.name.endswith('-routes')]
    if not files:
        click.echo('No OpenMap routes available')
        return

    with autopage.AutoPager() as out:
        for d in files:
            out.write(f"{d.name} ({humanize.naturalsize(sum(f.stat().st_size for f in d.iterdir()), binary=True)})")

@routes.command()
def delete() -> None:
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if d.name.endswith('-routes')]
    if not files:
        click.echo('No OpenMap routes to delete')
        return
    selected = select_multiple([f'{d.name} ({humanize.naturalsize(sum(f.stat().st_size for f in d.iterdir()), binary=True)})' for d in files], return_indices=True, pagination=True)
    if selected:
        click.confirm(f'Are you sure you want to delete {len(selected)} directory(ies)?', abort=True)
        freed = 0
        for i in selected:
            click.echo(f'Deleting {files[i]}')
            freed += sum(f.stat().st_size for f in files[i].iterdir())
            for f in files[i].iterdir():
                f.unlink()
            files[i].rmdir()
        click.echo(f'Freed {humanize.naturalsize(freed, binary=True)}')

# FIXME: document better
@cli.group()
def population() -> None:
    """Commands related to population data."""
    pass

@population.command()
def get() -> None:
    data_dir = Path(platformdirs.user_data_dir(package_name))
    data_dir.mkdir(parents=True, exist_ok=True)
    census_file = data_dir / 'ESTAT_Census_2021.feather'
    if census_file.exists():
        click.echo(f'Population density data already available in {data_dir}')
        return
    click.echo(f'Downloading population density data into {data_dir}')
    download_population_density(census_file)

@population.command()
def delete() -> None:
    data_dir = Path(platformdirs.user_data_dir(package_name))
    census_file = data_dir / 'ESTAT_Census_2021.feather'
    if census_file.exists():
        click.confirm(f'Are you sure you want to delete the file {census_file.name} ({humanize.naturalsize(census_file.stat().st_size, binary=True)})?', abort=True)
        freed = census_file.stat().st_size
        census_file.unlink()
        click.echo(f'Freed {humanize.naturalsize(freed, binary=True)}')
    else:
        click.echo('No population density data to delete')

@cli.group()
def administrative() -> None:
    """Commands related to administrative data."""
    pass

@administrative.command()
@click.option('--level', '-l', type=int, default=2)
def get(level : int) -> None:
    data_dir = Path(platformdirs.user_data_dir(package_name))
    administrative_file = data_dir / f'gadm41_EU28_level_{level}.feather'
    if administrative_file.exists():
        click.echo(f'Administrative data already available in {data_dir}')
        return
    click.echo(f'Downloading administrative data (gadm4.1 level {level}) into {data_dir}')
    download_gadm_administrative_data(administrative_file, level=level)

@administrative.command()
@click.option('--level', '-l', type=int, default=2)
def delete(level : int) -> None:
    data_dir = Path(platformdirs.user_data_dir(package_name))
    administrative_file = data_dir / f'gadm41_EU28_level_{level}.feather'
    if administrative_file.exists():
        click.confirm(f'Are you sure you want to delete the file {administrative_file.name} ({humanize.naturalsize(administrative_file.stat().st_size, binary=True)})?', abort=True)
        freed = administrative_file.stat().st_size
        administrative_file.unlink()
        click.echo(f'Freed {humanize.naturalsize(freed, binary=True)}')
    else:
        click.echo('No administrative data to delete')

# FIXME: document better
@cli.group()
def generate() -> None:
    """Commands related to instance generation."""
    pass

# TODO: remove defaults of city, radius, patients and departing points
@generate.command()
@click.argument('config-file', type=click.File())
@click.argument('cost-config-file', type=click.File())
@click.option('--instance-name', '-ni', type=str, required=False, prompt='Name of the instance')
@click.option('--name-prefix', '-np', type=str, required=False, prompt='Name prefix')
@click.option('--name-suffix', '-ns', type=str, required=False)
@click.option('--city', '-c', type=str, required=False, default='Udine')
@click.option('--radius', '-r', type=float, required=False, default=20.0)
@click.option('--patients', '-p', type=int, required=True, prompt='Number of patients', default=10)
@click.option('--departing-points', '-d', type=int, required=True, prompt='Number of departing', default=2)
@click.option('--arrival-points', '-a', type=int, required=False, prompt='Number of arrival', default=0)
@click.option('--arrival-point-type', '-at', type=click.Choice(('same_as_departure', 'shuffle_departures', 'different_than_departure')), default='same_as_departure')
@click.option('--sync-rates', '-sr', type=(float, float, float, float), default=(0.25, 0.25, 0.25, 0.25), help="Rate of single services, double independent services, double simultaneous services, double sequential services", required=False)
@click.option('--horizon', '-h', type=int, required=True, default=600)
@click.option('--incompatibility', '-i', type=float, default=0.1, prompt='Incompatibility rate')
@click.option('--service-classes', '-sc', type=(int, int), default=(3, 3))
@click.option('--time-window-met', '-twm', type=click.Choice(['at_service_start', 'at_service_end']), default='at_service_start')
@click.option('--optional-rate', '-or', type=float, required=False, default=0.0)
@click.option('--double-time-window-rate', '-dtwr', type=float, required=False, default=0.0)
@click.option('--preferred-caregiver-rate', '-pcr', type=float, required=False, default=0.0)
@click.option('--random-seed', '-rs', type=int, required=False, default=42)
@click.option('--no-intersect-administrative', '-n', is_flag=True, default=False)
def instance(config_file : TextIO,
             cost_config_file : TextIO,
             instance_name : str | None,
             name_prefix : str | None, 
             name_suffix : str | None,
             city : str, 
             radius : float, 
             patients : int, 
             departing_points : int, 
             arrival_points : int | None, 
             arrival_point_type : str, 
             sync_rates : tuple[float, float, float, float], 
             horizon : int, 
             incompatibility : float, 
             service_classes : tuple[int, int], 
             time_window_met : Literal['at_service_start', 'at_service_end'], 
             optional_rate : float,
             double_time_window_rate : float,
             preferred_caregiver_rate : float,
             random_seed : int,
             no_intersect_administrative : bool,
    )  -> None:
    radius = np.round(float(radius), 1)
    np.random.seed(random_seed)
    random.seed(random_seed)
    config = yaml.safe_load(config_file)
    config_file.close()
    cost_config = yaml.safe_load(cost_config_file)
    cost_config_file.close() 
    assert math.isclose(sum(sync_rates), 1.0), f"The sync-rates should sum up to 1.0 but found {sum(sync_rates)}"
    assert 0.0 <= incompatibility <= 1.0, "The incompatibility should be between 0.0 and 1.0 (included)" 
    assert arrival_point_type != 'different_than_departure' or (arrival_points is not None and arrival_points > 0), "When arrival points are different than departure ones, you should specify their number"
    assert 0.0 <= optional_rate <= 1.0, "The rate of optional patients should be between 0.0 and 1.0 (included)" 
    assert 0.0 <= double_time_window_rate <= 1.0, "The rate of double time window patients should be between 0.0 and 1.0 (included)" 
    assert 0.0 <= preferred_caregiver_rate <= 1.0, "The preferred caregiver rate should be between 0.0 and 1.0 (included)"
    if instance_name is not None and name_prefix is not None:
        click.secho(f"Since instance_name is given ({instance_name}), name_prefix ({name_prefix}) will be ignored.", fg="yellow")
        name_prefix = None
    data_dir = Path(platformdirs.user_data_dir(package_name))
    openmap_file = None
    if not city:
        if click.confirm("No city provided, would you like to select among the available files", default=True):
            files = [d for d in data_dir.iterdir() if '.osm' in d.suffixes]
            if not files:
                click.echo('No OpenMap data to delete')
                return
            click.secho("Select one: ", fg='magenta')
            selected = select([f'{d.name} ({humanize.naturalsize(d.stat().st_size, binary=True)})' for d in files], return_index=True, pagination=True)
            selected_file = files[selected]
            match = re.match(r"(.+)-\d+(?:\.\d+)?km", selected_file.name)
            if match:
                city = match.group(1)
        if not city:
            city = click.prompt("Enter the city name")
    
    click.secho(f"City {city}", fg='cyan')    
    # Geocode the city
    res = gpd.tools.geocode(city, provider='nominatim', user_agent=package_name)
    # The CRS of the result is WGS 84 (EPSG:4326), so we can set it explicitly
    res.set_crs(epsg=4326, inplace=True)
    click.echo(f'Area {city} centered at {res.iloc[0].address} ({res.iloc[0].geometry}), with a radius of {radius}km')

    # Find an appropriate CRS for the UTM zone
    crs = find_appropriate_crs(res.iloc[0].geometry.y, res.iloc[0].geometry.x)
    # Create a GeoDataFrame with the bounding box, with the appropriate CRS
    bbox = res.to_crs(crs).buffer(radius * 1000)
    bounds = sp.box(*bbox.to_crs(epsg=4326).total_bounds)

    # search for the openmap data file that contains the area
    openmap_file = None
    for file in data_dir.iterdir():
        if '.osm' not in file.suffixes:
            continue
        file_bbox = find_osm_bounds(file)
        if file_bbox.buffer(1E-6).contains(bounds):
            openmap_file = file
            break
    if openmap_file is None:
        click.echo(f'No OpenMap data available for {city}')
        if click.confirm('Would you like to download it now?', default=True):
            assert area.commands.get('get') is not None
            _callable = area.commands.get('get')
            assert _callable is not None
            _callable(city.lower(), radius, None, False, False)
            openmap_file = data_dir / f'{city}-{radius}km.osm.bz2'
        else:
            click.secho('No instance created.')
            return
    click.echo(f'OpenMap data available in {openmap_file}')
    
    # Search for the routes data
    routes_dir = None
    if '.bz2' in openmap_file.suffixes:
        ref_file_name = ".".join(openmap_file.name.split('.')[:-2])
    else:
        ref_file_name = ".".join(openmap_file.name.split('.')[:-1])
    for file in data_dir.iterdir():        
        if file.is_dir() and (file.name.startswith(ref_file_name)) and file.name.endswith('-routes'):
            routes_dir = file
            break

    if routes_dir is None:
        click.echo('No routes data available for this area, processing it now')
        prepare_osrm_data(openmap_file)
        routes_dir = data_dir / f'{ref_file_name}-routes'
    routes_file = routes_dir / (ref_file_name + '.osrm')
    click.echo(f'Routes data available in {routes_dir}')
    router = pyosrm.PyOSRM(str(routes_file), algorithm='MLD')

    population_density_file = data_dir / 'ESTAT_Census_2021.feather'
    if not population_density_file.exists():
        click.echo('No population density data available')
        # click.prompt('Would you like to download it now?', type=bool, default=True)
        click.echo('Downloading population data')
        download_population_density(population_density_file)  
    population_density = gpd.read_feather(population_density_file)
    
    # search for the closest administrative unit
    if not no_intersect_administrative:
        administrative_data_file = data_dir / 'gadm41_EU28_level_2.feather'
        if not administrative_data_file.exists():
            click.echo('No administrative data available')
            # click.prompt('Would you like to download it now?', type=bool, default=True)
            click.echo('Downloading administrative data')
            _callable = administrative.commands.get('get')
            assert _callable is not None
            _callable(2)
        administrative_data = gpd.read_feather(administrative_data_file)
        administrative_data = administrative_data.to_crs(res.crs)
        administrative_data = administrative_data[administrative_data.buffer(1E-6).contains(res.iloc[0].geometry)]  
        assert administrative_data.shape[0] == 1, 'The city should be contained in a single administrative unit'       
        population_density = population_density.to_crs(administrative_data.crs)
        population_density = population_density[population_density.geometry.intersects(administrative_data.iloc[0].geometry)]
   
    # Make the datasets compatible w.r.t. crs
    population_density.to_crs(crs, inplace=True)
    res.to_crs(crs, inplace=True)
    # Select the population density within the radius
    population_density = population_density[population_density.geometry.distance(res.iloc[0].geometry) < radius * 1000]
    # Generate the instance
    click.echo('Generating instance')       
    if arrival_point_type != 'different_than_departure':
        if arrival_points != 0:
            click.secho(f"When the arrival points are the same as departure or shuffled, no new arrival point will be created, found instead {arrival_points}", fg='yellow')
        arrival_points = 0
    assert arrival_points is not None
    locations, distances = generate_spatial(departing_points + patients + arrival_points, population_density, router)
    click.echo('Spatial distribution generated')
    assert locations is not None
    assert distances is not None

    generated_departing_points, generated_arrival_points, generated_caregivers, generated_patients, generated_services = generate_temporal(
        distances=distances,
        departing_locations=locations[:departing_points], 
        departing_indexes=[i for i in range(departing_points)],
        patient_locations=locations[departing_points:departing_points + patients],
        patient_indexes=[i for i in range(departing_points, departing_points + patients)],
        arrival_locations=locations[departing_points + patients:],
        arrival_indexes=[i for i in range(departing_points + patients, len(locations))],
        arrival_point_type=arrival_point_type,
        single=sync_rates[0], independent=sync_rates[1],  simultaneous=sync_rates[2], sequential=sync_rates[3], 
        horizon=horizon, 
        incompatibility=incompatibility,
        service_classes=service_classes,
        optional_rate=optional_rate,
        double_time_window_rate=double_time_window_rate,
        preferred_caregiver_rate=preferred_caregiver_rate,
        config=config
    )
    
    # FIXME: add the bounding box of the area and pass it to the instance for creation
    generator_infos = GeneratorInformation(
        city=city,
        radius=radius,
        patients=patients,
        departing_points=departing_points,
        arrival_points=arrival_points,
        arrival_point_type=arrival_point_type,
        sync_rates=sync_rates,
        incompatibility=incompatibility,
        service_classes=service_classes,
        optional_rate=optional_rate,
        double_time_window_rate=double_time_window_rate,
        no_intersect_administrative=no_intersect_administrative,
        patient_config=PatientGeneratorInformation(**config.get('patient')),
        caregiver_config=CaregiverGeneratorInformation(**config.get('caregiver')),
        preferred_caregiver_rate=preferred_caregiver_rate,
        random_seed=random_seed
    )
    if instance_name is None:
        instance_name = f'{name_prefix}-{city}-r{radius}-p{len(generated_patients)}-s{len(generated_services)}-c{len(generated_caregivers)}'
    if name_suffix:
        instance_name += f"-{name_suffix}"
    metadata = MetaData(
        time_window_met=time_window_met, 
        name=instance_name,
        origin="generated",
        horizon=horizon,
        generator_info=generator_infos,
        cost_components=CostComponents(**cost_config)
    )

    terminal_points : set = set()
    for point in set(generated_arrival_points):
        terminal_points.add(point)
    for point in set(generated_departing_points):
        terminal_points.add(point)

    instance = Instance( 
        metadata = metadata,
        distances=distances.tolist(),    
        terminal_points=terminal_points,
        caregivers=generated_caregivers,
        patients=generated_patients,
        services=generated_services,
    )
    if  config.get('caregiver', {}).get('lunch_break') is not None:
        lunch_break_time_window = config.get('caregiver', {}).get('lunch_break', {}).get('time_window')
        lunch_break_duration = config.get('caregiver', {}).get('lunch_break', {}).get('duration')   
        instance.lunch_breaks = LunchBreakSpec(start=lunch_break_time_window[0], end=lunch_break_time_window[1], min_duration=lunch_break_duration)

    with open(f"{instance.metadata.name}.json", "w") as f:
        f.write(instance.model_dump_json(indent=2, exclude_none=True))

    