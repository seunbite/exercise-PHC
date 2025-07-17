__all__ = ['flags', 'summation']

class Flags(object):
    def __init__(self, items):
        for key, val in items.items():
            setattr(self,key,val)

flags = Flags({
    'test': False, 
    'debug': False,
    "real_traj": False,
    "im_eval": False,
    "follow": False,
    "fixed": False,
    "divide_group": False,
    "no_collision_check": False,
    "fixed_path": False,
    "real_path": False,
    "small_terrain": False,
    "show_traj": False,
    "server_mode": False,
    "slow": False,
    "no_virtual_display": False,
    "render_o3d": False,
    "add_proj" : True
    })
