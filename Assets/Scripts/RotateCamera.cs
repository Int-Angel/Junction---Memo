using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RotateCamera : MonoBehaviour
{
    [SerializeField] private Transform target;
    [SerializeField] private float speed;

    // Start is called before the first frame update
    void Start()
    {
        transform.LookAt(target.transform.position);
    }

    // Update is called once per frame
    void Update()
    {
        transform.RotateAround(target.transform.position, new Vector3(0.0f, 1.0f, 0.0f), 20 * Time.deltaTime * speed);
    }
}
