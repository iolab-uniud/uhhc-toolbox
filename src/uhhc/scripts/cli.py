import click
from importlib import metadata

# import warnings
# warnings.simplefilter("always")

commands = []

try:
    from ..generator.cli import cli as cli_generator
except Exception:
    from ..generator._fallback_cli import cli as cli_generator
finally:
    commands.append(cli_generator)

from ..validator.cli import cli as cli_validator
commands.append(cli_validator)


def _print_version(ctx, param, value):
    """Eager callback to print version and exit."""
    if not value or ctx.resilient_parsing:
        return
    try:
        # Replace 'yourpackage' with the actual top-level package/distribution name
        ver = metadata.version("uhhc-toolbox")
    except metadata.PackageNotFoundError:
        from .. import __version__ as ver
    click.echo(ver)
    ctx.exit()


def main_cli():
    version_option = click.Option(
        ["--version", "-V"],
        is_flag=True,
        expose_value=False,
        is_eager=True,
        help="Show the version and exit.",
        callback=_print_version,
    )

    cli = click.CommandCollection(
        sources=commands,
        params=[version_option],  # attach the global --version
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    cli(obj={})
