using Unity.MLAgents;
using UnityEngine;

using Vector3 = UnityEngine.Vector3;

public class EnvironmentState
{
    // Singleton Construction
    private static EnvironmentState _instance = new EnvironmentState();
    private EnvironmentState()
    {
    }

    public static EnvironmentState Instance => _instance;
    // Environment Parameters are delivered to Academy by Environments SideChannel
    private readonly EnvironmentParameters _envParameters = Academy.Instance.EnvironmentParameters;
    
    /* Actual Environment State */
    
    // Position of the Target 
    public Vector3 TargetPosition
    {
        get
        {
            Vector3 targetPosition = new Vector3(0, 0, 0);
            targetPosition.y = 0.2f;// above floor
            targetPosition.x = _envParameters.GetWithDefault("target_position_x", float.NaN);
            targetPosition.z = _envParameters.GetWithDefault("target_position_z", float.NaN);

            // handle NaNs to allow for gameplay without Sidechannel
            // uniform randomization
            if (float.IsNaN(targetPosition.x) || float.IsNaN(targetPosition.z))
            {
                Debug.Log("TargetPosition was not set by SideChannel, creating random position");
                targetPosition.x = Random.value * 8 - 4;
                targetPosition.z = Random.value * 8 - 4;
            }
            return targetPosition;
        }
    }

    
}