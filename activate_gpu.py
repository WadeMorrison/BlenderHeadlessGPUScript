import bpy

def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()

    # Printing the devices to understand the structure
    print(cycles_preferences.devices)

    # If devices are not separated into CUDA and OpenCL, handle them in a unified way
    devices = cycles_preferences.devices

    if device_type == "CUDA":
        print("Setting to use CUDA Devices")
    elif device_type == "OPENCL":
        print("Setting to use OpenCL Devices")
    else:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []

    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        elif device.type == device_type:
            device.use = True
            activated_gpus.append(device.name)
        else:
            device.use = False

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    print(f"{len(activated_gpus)} devices detected: {', '.join(activated_gpus)}")
    return activated_gpus

# Enable CUDA GPUs
enable_gpus("CUDA")
