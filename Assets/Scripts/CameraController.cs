using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    [SerializeField] private Camera cam;
    [SerializeField] private Transform character;
    [SerializeField] private Transform village;
    [SerializeField] private Button button;

    Vector3 previousPosition;
    int currentState = 0;
    Transform target;

    private void Awake()
    {
        target = character;
        currentState = 0;
    }
    // Start is called before the first frame update
    void Start()
    {
        button.onClick.AddListener(ToggleVillageAndMainScreen);
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            previousPosition = cam.ScreenToViewportPoint(Input.mousePosition);
        }

        if (Input.GetMouseButton(0))
        {
            Vector3 direction = previousPosition - cam.ScreenToViewportPoint(Input.mousePosition);

            cam.transform.position = target.position;

            cam.transform.Rotate(new Vector3(1f, 0, 0), direction.y * 180);
            cam.transform.Rotate(new Vector3(0, 1f, 0), -direction.x * 180, Space.World);
            cam.transform.Translate(new Vector3(0, 0, -10));

            previousPosition = cam.ScreenToViewportPoint(Input.mousePosition);
        }
    }

    void ToggleVillageAndMainScreen()
    {
        if(currentState == 0)
        {
            currentState = 1;
            target = village;

            previousPosition = cam.ScreenToViewportPoint(Input.mousePosition);
            Vector3 direction = previousPosition - cam.ScreenToViewportPoint(Input.mousePosition);

            cam.transform.position = target.position;

            cam.transform.Rotate(new Vector3(1f, 0, 0), direction.y * 180);
            cam.transform.Rotate(new Vector3(0, 1f, 0), -direction.x * 180, Space.World);
            cam.transform.Translate(new Vector3(0, 0, -10));

            previousPosition = cam.ScreenToViewportPoint(Input.mousePosition);
        }
        else
        {
            currentState = 0;
            target = character;

            previousPosition = cam.ScreenToViewportPoint(Input.mousePosition);
            Vector3 direction = previousPosition - cam.ScreenToViewportPoint(Input.mousePosition);

            cam.transform.position = target.position;

            cam.transform.Rotate(new Vector3(1f, 0, 0), direction.y * 180);
            cam.transform.Rotate(new Vector3(0, 1f, 0), -direction.x * 180, Space.World);
            cam.transform.Translate(new Vector3(0, 0, -10));

            previousPosition = cam.ScreenToViewportPoint(Input.mousePosition);
        }
    }
}
