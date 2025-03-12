package fr.istic.web.rest;

import static jakarta.ws.rs.core.UriBuilder.fromPath;

import fr.istic.service.ResponseGroupService;
import fr.istic.web.rest.errors.AccountResourceException;
import fr.istic.web.rest.errors.BadRequestAlertException;
import fr.istic.web.util.HeaderUtil;
import fr.istic.web.util.ResponseUtil;
import fr.istic.service.dto.ResponseGroupDTO;

import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import fr.istic.domain.Authority;
import fr.istic.domain.ResponseGroup;
import fr.istic.domain.User;
import fr.istic.security.AuthoritiesConstants;
import fr.istic.service.Paged;
import fr.istic.service.SecurityService;
//import fr.istic.service.customdto.ResponseGroupsIdsDto;
import fr.istic.web.rest.vm.PageRequestVM;
import fr.istic.web.rest.vm.SortRequestVM;
import fr.istic.web.util.PaginationUtil;

import jakarta.annotation.security.RolesAllowed;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * REST controller for managing {@link fr.istic.domain.ResponseGroup}.
 */
@Path("/api/responseGroups")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@ApplicationScoped
public class ResponseGroupResource {

    private final Logger log = LoggerFactory.getLogger(ResponseGroupResource.class);

    private static final String ENTITY_NAME = "responseGroup";

    @ConfigProperty(name = "application.name")
    String applicationName;

    @Inject
    SecurityService securityService;

    @Inject
    ResponseGroupService responseGroupService;

    /**
     * {@code POST  /responseGroups} : Create a new responseGroup.
     *
     * @param responseGroupDTO the responseGroupDTO to create.
     * @return the {@link Response} with status {@code 201 (Created)} and with body the new responseGroupDTO, or with status {@code 400 (Bad Request)} if the responseGroup already has an ID.
     */
    @POST
    @RolesAllowed({AuthoritiesConstants.USER, AuthoritiesConstants.ADMIN})
    public Response createResponseGroup(ResponseGroupDTO responseGroupDTO, @Context UriInfo uriInfo) {
        log.debug("REST request to save ResponseGroup : {}", responseGroupDTO);
        if (responseGroupDTO.id != null) {
            throw new BadRequestAlertException("A new responseGroup cannot already have an ID", ENTITY_NAME, "idexists");
        }
        var result = responseGroupService.persistOrUpdate(responseGroupDTO);
        var response = Response.created(fromPath(uriInfo.getPath()).path(result.id.toString()).build()).entity(result);
        HeaderUtil.createEntityCreationAlert(applicationName, true, ENTITY_NAME, result.id.toString()).forEach(response::header);
        return response.build();
    }

    /**
     * {@code PUT  /responseGroups} : Updates an existing responseGroup.
     *
     * @param responseGroupDTO the responseGroupDTO to update.
     * @return the {@link Response} with status {@code 200 (OK)} and with body the updated responseGroupDTO,
     * or with status {@code 400 (Bad Request)} if the responseGroupDTO is not valid,
     * or with status {@code 500 (Internal Server Error)} if the responseGroupDTO couldn't be updated.
     */
    @PUT
    @RolesAllowed({AuthoritiesConstants.USER, AuthoritiesConstants.ADMIN})
    public Response updateResponseGroup(ResponseGroupDTO responseGroupDTO, @Context SecurityContext ctx) {
        log.debug("REST request to update ResponseGroup : {}", responseGroupDTO);
        if (responseGroupDTO.id == null) {
            throw new BadRequestAlertException("Invalid id", ENTITY_NAME, "idnull");
        }
        if (!securityService.canAccess(ctx, responseGroupDTO.id, ResponseGroup.class)) {
            return Response.status(403, "Current user cannot access to this resource").build();
        }
        var result = responseGroupService.persistOrUpdate(responseGroupDTO);
        var response = Response.ok().entity(result);
        HeaderUtil.createEntityUpdateAlert(applicationName, true, ENTITY_NAME, responseGroupDTO.id.toString()).forEach(response::header);
        return response.build();
    }

    /**
     * {@code DELETE  /responseGroups/:id} : delete the "id" responseGroup.
     *
     * @param id the id of the responseGroupDTO to delete.
     * @return the {@link Response} with status {@code 204 (NO_CONTENT)}.
     */
    @DELETE
    @Path("/{id}")
    @RolesAllowed({AuthoritiesConstants.USER, AuthoritiesConstants.ADMIN})
    public Response deleteResponseGroup(@PathParam("id") Long id, @Context SecurityContext ctx) {
        log.debug("REST request to delete ResponseGroup : {}", id);
        try {
            // Security check
            if (!securityService.canAccess(ctx, id, ResponseGroup.class)) {
                log.error("User is not authorized to delete ResponseGroup with id: {}", id);
                return Response.status(403, "Current user cannot access this resource").build();
            }

            // Attempt deletion
            responseGroupService.delete(id);
            log.info("ResponseGroup with id {} deleted successfully", id);
        } catch (Exception e) {
            log.error("Failed to delete ResponseGroup with id {}: {}", id, e.getMessage(), e);
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity("Failed to delete ResponseGroup with id: " + id + ". Error: " + e.getMessage())
                    .build();
        }

        // If successful
        var response = Response.noContent();
        HeaderUtil.createEntityDeletionAlert(applicationName, true, ENTITY_NAME, id.toString()).forEach(response::header);
        return response.build();
    }


        /**
     * {@code DELETE  /responseGroups/question/:questionId} : delete the "id" responseGroup.
     *
     * @param id the questionId of the question associated to responseGroup to delete.
     * @return the {@link Response} with status {@code 204 (NO_CONTENT)}.
     */
    @DELETE
    @Path("/question/{questionId}")
    @RolesAllowed({AuthoritiesConstants.USER, AuthoritiesConstants.ADMIN})
    public Response deleteResponseGroupByQuestionId(@PathParam("questionId") Long id, @Context SecurityContext ctx) {
        log.debug("REST request to delete ResponseGroup : {}", id);
        try {
            // Security check
            if (!securityService.canAccess(ctx, id, ResponseGroup.class)) {
                log.error("User is not authorized to delete ResponseGroup with id: {}", id);
                return Response.status(403, "Current user cannot access this resource").build();
            }

            // Attempt deletion
            responseGroupService.deleteByQuestionId(id);
            log.info("ResponseGroup with id {} deleted successfully", id);
        } catch (Exception e) {
            log.error("Failed to delete ResponseGroup with id {}: {}", id, e.getMessage(), e);
            return Response.status(Response.Status.INTERNAL_SERVER_ERROR)
                    .entity("Failed to delete ResponseGroup with id: " + id + ". Error: " + e.getMessage())
                    .build();
        }

        // If successful
        var response = Response.noContent();
        HeaderUtil.createEntityDeletionAlert(applicationName, true, ENTITY_NAME, id.toString()).forEach(response::header);
        return response.build();
    }


    /**
     * {@code GET  /responseGroups} : get all the responseGroups.
     *
     * @param pageRequest the pagination information.
     * @return the {@link Response} with status {@code 200 (OK)} and the list of responseGroups in body.
     */
    @GET
    @RolesAllowed({AuthoritiesConstants.USER, AuthoritiesConstants.ADMIN})
    public Response getAllResponseGroups(@BeanParam PageRequestVM pageRequest, @BeanParam SortRequestVM sortRequest, @Context UriInfo uriInfo, @Context SecurityContext ctx) {
        log.debug("REST request to get a page of ResponseGroups");
        var page = pageRequest.toPage();
        var sort = sortRequest.toSort();
        MultivaluedMap<String, String> param = uriInfo.getQueryParameters();
        Paged<ResponseGroupDTO> result = new Paged<>(0, 0, 0, 0, new ArrayList<>());
        if (param.containsKey("questionId")) {
            List<String> questionId = param.get("questionId");
            result = responseGroupService.findResponseGroupByQuestionId(page, Long.parseLong(questionId.get(0)));
        } else {
            if (ctx.getUserPrincipal().getName() != null) {
                var userLogin = Optional.ofNullable(ctx.getUserPrincipal().getName());
                if (!userLogin.isPresent()) {
                    throw new AccountResourceException("Current user login not found");
                }
                var user = User.findOneByLogin(userLogin.get());
                if (!user.isPresent()) {
                    throw new AccountResourceException("User could not be found");

                } else if (user.get().authorities.size() >= 1 && user.get().authorities.stream().anyMatch(e1 -> e1.equals(new Authority("ROLE_USER")))) {
                    //Ici j'ai modif l'autorisation de Admin -> User, je sais pas si c'est bien ou pas mais voila ca me permet mon ajout
                    result = responseGroupService.findAll(page);
                } else {
                    return Response.status(403, "Current user cannot access to this resource").build();
                }
            }
        }
        var response = Response.ok().entity(result.content);
        response = PaginationUtil.withPaginationInfo(response, uriInfo, result);
        return response.build();
    }

    /**
     * {@code GET  /responseGroups/:id} : get the "id" responseGroup.
     *
     * @param id the id of the responseGroupDTO to retrieve.
     * @return the {@link Response} with status {@code 200 (OK)} and with body the responseGroupDTO, or with status {@code 404 (Not Found)}.
     */
    @GET
    @Path("/{id}")
    @RolesAllowed({AuthoritiesConstants.USER, AuthoritiesConstants.ADMIN})
    public Response getResponseGroup(@PathParam("id") Long id, @Context SecurityContext ctx) {
        log.debug("REST request to get ResponseGroup : {}", id);
        if (!securityService.canAccess(ctx, id, ResponseGroup.class)) {
            return Response.status(403, "Current user cannot access to this resource").build();
        }
        Optional<ResponseGroupDTO> responseGroupDTO = responseGroupService.findOne(id);
        return ResponseUtil.wrapOrNotFound(responseGroupDTO);
    }
}
