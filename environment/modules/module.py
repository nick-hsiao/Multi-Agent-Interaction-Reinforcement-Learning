
class EnvModule():
    '''
        Dummy class outline for "Environment Modules".
        NOTE: If in any function you are going to randomly sample a number,
            use env._random_state instead of numpy.random
    '''
    def build_world_step(self, world):
        '''
            Instantiate the world. Place modules components into placement grid.
            Args:
                world: the target World.
        '''
        return True

    def build_render(self, viewer):
        '''
            Each module handles rendering it's own compnents and populating the viewer.
            Args:
                viewer (rendering.Viewer): the viewer
        '''
        return True

    def observation_step(self, world):
        '''
            Create any observations specific to this module.
            Args:
                env (gym.env): the environment
        '''
        return {}
