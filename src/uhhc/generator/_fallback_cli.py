import click

# FIXME: 
@click.group()
def cli():
    """This is the main command group related to data files for geographic management and instance generation."""
    pass

class FallbackGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        @click.command(name=cmd_name)
        def dummy_command(*args, **kwargs):
            click.secho(
                f"⚠️ The command '{ctx.command_path} {cmd_name}' is not available "
                "because the generator functionality is not installed. You should install the hhcrsp-toolbox with the generator extra (requires osrm-backend library to be installed along).",
                fg="magenta"
            )
        return dummy_command

# FIXME: document better
@cli.group(cls=FallbackGroup)
def area():
    """Commands related to geographic data files management.
    
    ⚠️ Subcommands not available since the generator functionality is not installed. ⚠️
    """
    pass

# FIXME: document better using pyosrm
@cli.group(cls=FallbackGroup)
def routes():
    """
    Commands related to routing file management.

    ⚠️ Subcommands not available since the generator functionality is not installed. ⚠️
    """
    pass

# FIXME: document better
@cli.group(cls=FallbackGroup)
def population():
    """
    Commands related to population data.

    ⚠️ Subcommands not available since the generator functionality is not installed. ⚠️    
    """
    pass

@cli.group(cls=FallbackGroup)
def administrative():
    """
    Commands related to administrative data.

    ⚠️ Subcommands not available since the generator functionality is not installed. ⚠️ 
    """
    pass

# FIXME: document better
@cli.group(cls=FallbackGroup)
def generate():
    """
    Commands related to instance generation.

     ⚠️ Subcommands not available since the generator functionality is not installed. ⚠️
    """
    pass