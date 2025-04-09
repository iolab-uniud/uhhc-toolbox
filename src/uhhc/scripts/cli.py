import click
import warnings
# warnings.simplefilter("always")

commands = []

try:
    from .. generator.cli import cli as cli_generator
except:
    from ..generator._fallback_cli import cli as cli_generator
finally:
    commands.append(cli_generator)

from .. validator.cli import cli as cli_validator
commands.append(cli_validator)
 
def main_cli():
    cli = click.CommandCollection(sources=commands)
    cli(obj={})
